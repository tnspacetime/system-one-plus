"""Inference-only static KV storage and device-side cursors for proposal rows."""

import torch
from transformers.cache_utils import Cache


class DecodeCache(Cache):
    """Physical cache slots and logical RoPE positions are intentionally separate."""

    def __init__(self, parents, budgets):
        super().__init__(layers=[])
        if torch.is_grad_enabled():
            raise RuntimeError("DecodeCache is inference-only")
        lengths = [parent.get_seq_length() for parent in parents]
        self.capacity = max(length + budget for length, budget in zip(lengths, budgets, strict=True))
        self.used_upper = max(lengths)
        device = parents[0].segments[0].layers[0][0].device
        self.lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
        self.active = torch.ones(len(parents), device=device, dtype=torch.bool)
        self.attention_lengths = self.lengths
        self.storage = []
        # Native FlashAttention cache order: batch, sequence, KV heads, head dim.
        for layer in range(len(parents[0].segments[0].layers)):
            example = parents[0].segments[0].layers[layer][0]
            shape = (len(parents), self.capacity, example.shape[1], example.shape[-1])
            keys, values = example.new_zeros(shape), example.new_zeros(shape)
            for row, parent in enumerate(parents):
                offset = 0
                for segment in parent.segments:
                    k, v = segment.layers[layer]
                    keys[row : row + 1, offset : offset + segment.length].copy_(k.transpose(1, 2))
                    values[row : row + 1, offset : offset + segment.length].copy_(v.transpose(1, 2))
                    offset += segment.length
            self.storage.append((keys, values))

    def get_seq_length(self, layer_idx=0):
        return self.used_upper

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[-2] != 1:
            raise ValueError("static decoding requires one token")
        keys, values = self.storage[layer_idx]
        rows = torch.arange(self.lengths.numel(), device=keys.device)
        # Completed rows may stay resident until the next batched status check.
        # They reuse their last slot and never consume capacity or affect other rows.
        slots = self.lengths.long().clamp(max=self.capacity - 1)
        keys[rows, slots] = key_states[:, :, 0, :].to(keys.dtype)
        values[rows, slots] = value_states[:, :, 0, :].to(values.dtype)
        return keys[:, : self.used_upper].transpose(1, 2), values[:, : self.used_upper].transpose(1, 2)

    def retain(self, rows):
        indices = torch.as_tensor(rows, device=self.lengths.device, dtype=torch.long)
        self.storage = [(k.index_select(0, indices), v.index_select(0, indices)) for k, v in self.storage]
        self.lengths = self.lengths.index_select(0, indices)
        self.active = self.active.index_select(0, indices)
        self.attention_lengths = self.attention_lengths.index_select(0, indices)

    def reference_mask(self):
        return (
            torch.arange(self.used_upper, device=self.lengths.device)[None, :]
            < self.attention_lengths[:, None]
        )[:, None, None, :]

    def step(self, backbone, tokens, positions, device, active=None):
        if torch.is_grad_enabled():
            raise RuntimeError("decoding must run without autograd")
        tokens = torch.as_tensor(tokens, device=device, dtype=torch.long)
        positions = torch.as_tensor(positions, device=device, dtype=torch.long)
        self.active = torch.ones_like(self.lengths, dtype=torch.bool) if active is None else active
        self.attention_lengths = self.lengths + self.active.to(torch.int32)
        self.used_upper = min(self.capacity, self.used_upper + 1)
        output = backbone(
            input_ids=tokens[:, None],
            position_ids=torch.where(self.active, positions, 0)[:, None],
            cache_position=torch.zeros(1, device=device, dtype=torch.long),
            attention_mask={"full_attention": None},
            sop_decode=self,
            past_key_values=self,
            use_cache=False,
            return_dict=True,
        )
        self.lengths = self.attention_lengths
        return output.last_hidden_state[:, 0]
