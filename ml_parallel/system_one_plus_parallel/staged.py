"""Differentiable prefix sharing and length-bucketed attention continuations.

Persistent caches contain only each node's own KV tensors. Branches refer to the
same immutable prefix segments. Attention receives a temporary contiguous KV batch;
there is no request-wide attention matrix or mutable inference cache in training.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface
from transformers.cache_utils import Cache

from .attention import BranchLayout, flash_branches, flash_decode, reference_attention, use_flash

# Preserve the registered name for older checkpoint metadata; CUDA dispatch is
# explicit FlashAttention, not PyTorch's opportunistic SDPA selection.
ATTENTION_BACKEND = "sop_staged_sdpa"
EXECUTION_VERSION = "staged_flash_v3"


def staged_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """CUDA uses explicit Flash kernels; portable/reference execution uses SDPA."""
    layout, cache = kwargs.get("sop_layout"), kwargs.get("sop_decode")
    mode = getattr(getattr(module, "config", None), "sop_attention_mode", "auto")
    if use_flash(query.device, mode):
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("FlashAttention received FP32 queries; enable the configured compute dtype")
        if attention_mask is not None:
            raise RuntimeError("FlashAttention requires length metadata, not an explicit attention mask")
        if cache is not None:
            output = flash_decode(query, cache, module.layer_idx, scaling=scaling)
        elif layout is not None:
            output = flash_branches(query, key, value, layout, dropout=dropout, scaling=scaling)
        else:
            raise RuntimeError("missing FlashAttention branch/decode metadata")
    else:
        if layout is not None:
            attention_mask = layout.reference_mask(query.device)
        elif cache is not None:
            attention_mask = cache.reference_mask()
        output = reference_attention(query, key, value, attention_mask, dropout=dropout, scaling=scaling)
    return output, None


AttentionInterface.register(ATTENTION_BACKEND, staged_attention)


@dataclass(frozen=True, eq=False)
class KVSegment:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    @property
    def length(self):
        return self.layers[0][0].shape[-2]


@dataclass(frozen=True, eq=False)
class BranchCache:
    segments: tuple[KVSegment, ...] = ()

    def get_seq_length(self):
        return sum(segment.length for segment in self.segments)

    def append(self, segment):
        return BranchCache((*self.segments, segment))

    def prefix(self, length):
        if not 0 <= length <= self.get_seq_length():
            raise ValueError("invalid cache prefix length")
        selected = []
        for segment in self.segments:
            if not length:
                break
            if length >= segment.length:
                selected.append(segment)
                length -= segment.length
            else:
                selected.append(
                    KVSegment(tuple((k[..., :length, :], v[..., :length, :]) for k, v in segment.layers))
                )
                break
        return BranchCache(tuple(selected))

    def layer(self, index):
        if len(self.segments) == 1:
            return self.segments[0].layers[index]
        return tuple(
            torch.cat([segment.layers[index][kind] for segment in self.segments], dim=-2) for kind in (0, 1)
        )

    def to_legacy_cache(self):
        """Materialize for diagnostics only; execution keeps the segment references."""
        return tuple(self.layer(i) for i in range(len(self.segments[0].layers))) if self.segments else ()


class _ContinuationCache(Cache):
    """Ephemeral per-forward collector; update never mutates a parent or appends twice."""

    def __init__(self, parents, layers):
        super().__init__(layers=[])
        self.parents = parents
        self.current = [None] * layers

    def get_seq_length(self, layer_idx=0):
        return self.parents[0].get_seq_length()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.current[layer_idx] = (key_states, value_states)
        if not self.get_seq_length():
            return key_states, value_states
        if all(parent is self.parents[0] for parent in self.parents):
            keys, values = self.parents[0].layer(layer_idx)
            keys = keys.expand(len(self.parents), -1, -1, -1)
            values = values.expand(len(self.parents), -1, -1, -1)
        else:
            rows = [parent.layer(layer_idx) for parent in self.parents]
            width = max(parent.get_seq_length() for parent in self.parents)
            keys, values = (
                torch.cat([F.pad(row[kind], (0, 0, 0, width - row[kind].shape[-2])) for row in rows], dim=0)
                for kind in (0, 1)
            )
        return torch.cat((keys, key_states), dim=-2), torch.cat((values, value_states), dim=-2)


@dataclass(frozen=True)
class Continuation:
    tokens: tuple[int, ...]
    positions: tuple[int, ...]
    parent: BranchCache = BranchCache()


@dataclass(frozen=True)
class ContinuationOutput:
    hidden: torch.Tensor
    cache: BranchCache


def _run_batch(backbone, branches, *, device, checkpointing):
    lengths = [len(branch.tokens) for branch in branches]
    prefixes = [branch.parent.get_seq_length() for branch in branches]
    width = max(lengths)
    ids = torch.tensor([(*b.tokens, *((0,) * (width - len(b.tokens)))) for b in branches], device=device)
    positions = torch.tensor(
        [(*b.positions, *((0,) * (width - len(b.positions)))) for b in branches], device=device
    )
    layout = BranchLayout.build(prefixes, lengths, device)
    layers = backbone.config.num_hidden_layers
    # Pass unique prefix tensors explicitly to checkpoint. No detach/no_grad, and
    # no mutable cache survives checkpoint recomputation.
    segments, segment_indices, parent_paths = [], {}, []
    for branch in branches:
        path = []
        for segment in branch.parent.segments:
            if id(segment) not in segment_indices:
                segment_indices[id(segment)] = len(segments)
                segments.append(segment)
            path.append(segment_indices[id(segment)])
        parent_paths.append(tuple(path))
    tensors = tuple(tensor for segment in segments for pair in segment.layers for tensor in pair)

    def execute(input_ids, position_ids, *prefix_tensors):
        restored = [
            KVSegment(
                tuple(
                    (prefix_tensors[start + 2 * i], prefix_tensors[start + 2 * i + 1]) for i in range(layers)
                )
            )
            for start in range(0, len(prefix_tensors), 2 * layers)
        ]
        parents = {}
        for path in parent_paths:
            parents.setdefault(path, BranchCache(tuple(restored[i] for i in path)))
        cache = _ContinuationCache(tuple(parents[path] for path in parent_paths), layers)
        result = backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            attention_mask={"full_attention": None},
            sop_layout=layout,
            use_cache=False,
            return_dict=True,
        )
        return (result.last_hidden_state, *(tensor for pair in cache.current for tensor in pair))

    if checkpointing and torch.is_grad_enabled():
        result = checkpoint(execute, ids, positions, *tensors, use_reentrant=False)
    else:
        result = execute(ids, positions, *tensors)
    outputs = []
    for row, branch in enumerate(branches):
        segment = KVSegment(
            tuple(
                (
                    result[1 + 2 * i][row : row + 1, :, : lengths[row]],
                    result[2 + 2 * i][row : row + 1, :, : lengths[row]],
                )
                for i in range(layers)
            )
        )
        outputs.append(ContinuationOutput(result[0][row, : lengths[row]], branch.parent.append(segment)))
    return outputs


def run_continuations(backbone, branches, *, device, checkpointing=False, max_batch_size=16, bucket_width=0):
    """Optionally bucket nearby lengths with padding; zero preserves exact buckets."""
    if max_batch_size < 1 or bucket_width < 0:
        raise ValueError("invalid batch size or bucket width")
    groups = {}
    for index, branch in enumerate(branches):
        if not branch.tokens or len(branch.tokens) != len(branch.positions):
            raise ValueError("continuations need matching nonempty tokens and position IDs")
        if min(branch.positions) < 0 or max(branch.positions) >= backbone.config.max_position_embeddings:
            raise ValueError("continuation exceeds model position context")
        key = (branch.parent.get_seq_length(), len(branch.tokens))
        if bucket_width:
            # Keep empty parents separate: no cached layers exist for those rows.
            key = (bool(key[0]), *(length // bucket_width for length in key))
        groups.setdefault(key, []).append((index, branch))
    outputs = [None] * len(branches)
    for group in groups.values():
        for start in range(0, len(group), max_batch_size):
            rows = group[start : start + max_batch_size]
            results = _run_batch(
                backbone, [branch for _, branch in rows], device=device, checkpointing=checkpointing
            )
            for (index, _), result in zip(rows, results, strict=True):
                outputs[index] = result
    return outputs
