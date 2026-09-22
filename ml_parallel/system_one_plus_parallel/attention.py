"""Explicit FlashAttention CUDA kernels and a portable numerical reference path."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from itertools import accumulate

import torch
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right

FLASH_VERSION = "2.8.3"


@lru_cache(maxsize=1)
def flash_ops():
    try:
        installed = version("flash-attn")
        if installed.split("+", 1)[0] != FLASH_VERSION:
            raise RuntimeError(f"flash-attn=={FLASH_VERSION} is required; found {installed}")
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except ImportError as exc:
        raise RuntimeError("CUDA execution requires flash-attn==2.8.3; see README GPU setup") from exc
    return flash_attn_varlen_func, flash_attn_with_kvcache


def use_flash(device, mode="auto"):
    if mode not in {"auto", "flash", "reference"}:
        raise ValueError("attention_mode must be auto, flash, or reference")
    if mode == "reference":
        return False
    if mode == "flash" and torch.device(device).type != "cuda":
        raise ValueError("FlashAttention requires CUDA")
    return torch.device(device).type == "cuda"


def validate_flash(device):
    if torch.version.hip or torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("this FlashAttention runtime requires an NVIDIA Ampere or newer GPU")
    flash_ops()


def compute_dtype(device, requested="auto", attention_mode="auto"):
    device = torch.device(device)
    flash = use_flash(device, attention_mode)
    if requested == "auto":
        requested = "bf16" if flash else "fp32"
    if requested not in {"fp32", "bf16", "fp16"}:
        raise ValueError("unsupported compute dtype")
    if flash and requested == "fp32":
        raise ValueError(
            "FlashAttention requires bf16 or fp16; use reference explicitly for FP32 diagnostics"
        )
    if device.type != "cuda" and requested != "fp32":
        raise ValueError("the portable runtime uses fp32")
    if flash:
        validate_flash(device)
    if device.type == "cuda" and requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unavailable on this CUDA device")
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[requested]


@dataclass(frozen=True)
class BranchLayout:
    prefixes: tuple[int, ...]
    lengths: tuple[int, ...]
    q_indices: torch.Tensor
    kv_indices: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor

    @classmethod
    def build(cls, prefixes, lengths, device):
        pwidth, qwidth = max(prefixes), max(lengths)
        qi, ki = [], []
        for row, (prefix, length) in enumerate(zip(prefixes, lengths, strict=True)):
            qi.extend(range(row * qwidth, row * qwidth + length))
            start = row * (pwidth + qwidth)
            ki.extend(range(start, start + prefix))
            ki.extend(range(start + pwidth, start + pwidth + length))
        return cls(
            tuple(prefixes),
            tuple(lengths),
            torch.tensor(qi, device=device),
            torch.tensor(ki, device=device),
            torch.tensor([0, *accumulate(lengths)], device=device, dtype=torch.int32),
            torch.tensor(
                [0, *accumulate(p + q for p, q in zip(prefixes, lengths, strict=True))],
                device=device,
                dtype=torch.int32,
            ),
        )

    def reference_mask(self, device):
        if len(set(zip(self.prefixes, self.lengths))) == 1:
            return None
        pwidth, qwidth = max(self.prefixes), max(self.lengths)
        prefix = (
            torch.arange(pwidth, device=device)[None, :] < torch.tensor(self.prefixes, device=device)[:, None]
        )
        suffix = (
            torch.arange(qwidth, device=device)[None, :] < torch.tensor(self.lengths, device=device)[:, None]
        )
        causal = torch.arange(qwidth, device=device)[None, :] <= torch.arange(qwidth, device=device)[:, None]
        return torch.cat([prefix[:, None, :].expand(-1, qwidth, -1), suffix[:, None, :] & causal], -1)[
            :, None
        ]


def flash_branches(query, key, value, layout, *, dropout=0.0, scaling=None):
    """Unpad Q/K/V, run one differentiable varlen kernel, then restore row layout."""
    op, _ = flash_ops()
    batch, heads, width, dim = query.shape
    q = query.transpose(1, 2).reshape(-1, heads, dim).index_select(0, layout.q_indices)
    k = key.transpose(1, 2).reshape(-1, key.shape[1], dim).index_select(0, layout.kv_indices)
    v = value.transpose(1, 2).reshape(-1, value.shape[1], dim).index_select(0, layout.kv_indices)
    output = op(
        q,
        k,
        v,
        layout.cu_q,
        layout.cu_k,
        max(layout.lengths),
        max(p + n for p, n in zip(layout.prefixes, layout.lengths, strict=True)),
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    return (
        output.new_zeros(batch * width, heads, dim)
        .index_copy(0, layout.q_indices, output)
        .view(batch, width, heads, dim)
    )


def flash_decode(query, cache, layer_index, *, scaling=None):
    """Queries and cached keys already have Qwen's logical-position RoPE applied."""
    _, op = flash_ops()
    keys, values = cache.storage[layer_index]
    return op(
        query.transpose(1, 2),
        keys,
        values,
        cache_seqlens=cache.attention_lengths,
        softmax_scale=scaling,
        causal=False,
    )


def reference_attention(query, key, value, mask, *, dropout=0.0, scaling=None):
    qlength, klength = query.shape[-2], key.shape[-2]
    causal = mask is None and qlength == klength and qlength > 1
    if mask is None and 1 < qlength < klength:
        mask = causal_lower_right(qlength, klength)
    # Explicitly expand grouped KV heads only in the portable/reference backend.
    if query.shape[1] != key.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=causal, dropout_p=dropout, scale=scaling
    )
    return output.transpose(1, 2).contiguous()
