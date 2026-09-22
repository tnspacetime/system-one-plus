"""Finite-state structural constraints for action generation."""

import torch

from .errors import ProposalOutputError


class ActionGrammar:
    """Validate and decode a completed token sequence on the host."""

    def __init__(self, codec, limit):
        self.codec = codec
        self.limit = limit
        self.actions = []
        self.current = None
        self.finished = False
        self.tokens = []

    def consume(self, token):
        ids = self.codec.ids
        self.tokens.append(token)
        if self.current is None:
            if token == ids["done"]:
                self.finished = True
            elif token == ids["action"] and len(self.actions) < self.limit:
                self.current = []
            else:
                raise ProposalOutputError("invalid action boundary")
        elif token == ids["action_end"]:
            text = self.codec.tokenizer.decode(self.current, skip_special_tokens=True).strip()
            if not text:
                raise ProposalOutputError("empty generated action")
            self.actions.append(text)
            self.current = None
        elif token in self.codec.reserved:
            raise ProposalOutputError("control token inside action")
        else:
            self.current.append(token)


class BatchGrammar:
    """Batched grammar transitions, validity checks, and token selection on device.

    No tensor is read by Python in select(). Text is decoded only after a row
    finishes. A final parser verifies nonempty decoded actions, including Unicode
    whitespace assembled from several byte-level vocabulary pieces.
    """

    ACTIVE, DONE, NONFINITE, NO_TOKEN, EXHAUSTED = range(5)

    def __init__(self, codec, rows, vocab_size, device):
        self.codec = codec
        self.device = torch.device(device)
        self.limits = torch.tensor([r["count"] for r in rows], device=device)
        self.budgets = torch.tensor([r["budget"] for r in rows], device=device)
        self.temperature = torch.tensor([r["temperature"] for r in rows], device=device)[:, None]
        self.status = torch.zeros(len(rows), dtype=torch.long, device=device)
        self.inside = torch.zeros(len(rows), dtype=torch.bool, device=device)
        self.has_content = torch.zeros_like(self.inside)
        self.counts = torch.zeros_like(self.status)
        self.history = torch.full(
            (len(rows), max(r["budget"] for r in rows)), codec.ids["done"], device=device, dtype=torch.long
        )
        self.text_allowed = torch.arange(vocab_size, device=device) < len(codec.tokenizer)
        self.text_allowed[list(codec.reserved)] = False
        # Once per tokenizer, outside decoding: no repeated unfinished-text decode.
        if not hasattr(codec, "_content_ids"):
            codec._content_ids = []
            for start in range(0, len(codec.tokenizer), 4096):
                ids = range(start, min(start + 4096, len(codec.tokenizer)))
                texts = codec.tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=True)
                codec._content_ids.extend(i for i, text in zip(ids, texts, strict=True) if text.strip())
        self.content = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        self.content[codec._content_ids] = True
        self.generators = []
        self.rng_device = self.device if self.device.type != "mps" else torch.device("cpu")
        for row in rows:
            generator = torch.Generator(device=self.rng_device).manual_seed(row["seed"])
            self.generators.append(generator)
        self.sample = any(r["temperature"] > 0 for r in rows)
        vocab = torch.arange(vocab_size, device=device)
        self.is_action = vocab == codec.ids["action"]
        self.is_end = vocab == codec.ids["action_end"]
        self.is_done = vocab == codec.ids["done"]

    @property
    def active(self):
        return self.status == self.ACTIVE

    def select(self, logits, step):
        active = self.active
        outside_allowed = self.is_done[None, :] | (
            self.is_action[None, :] & (self.counts < self.limits)[:, None]
        )
        inside_allowed = self.text_allowed[None, :] | (self.is_end[None, :] & self.has_content[:, None])
        allowed = torch.where(self.inside[:, None], inside_allowed, outside_allowed)
        finite = torch.isfinite(logits).all(-1)
        scores = logits.masked_fill(~allowed, -torch.inf)
        valid = torch.isfinite(scores).any(-1)
        healthy = active & finite & valid
        # Invalid/finished rows select a harmless boundary without invalid softmax.
        scores = torch.where(healthy[:, None], scores, torch.where(self.is_done[None, :], 0.0, -torch.inf))
        token = scores.argmax(-1)
        if self.sample:
            # Per-search device RNGs preserve streams across batching/compaction.
            uniforms = torch.stack(
                [torch.rand((), device=self.rng_device, generator=g) for g in self.generators]
            ).to(self.device)
            cdf = torch.softmax(scores / self.temperature.clamp_min(1e-6), -1).cumsum(-1)
            cdf = cdf / cdf[:, -1:]
            sampled = (
                torch.searchsorted(cdf.contiguous(), uniforms[:, None].contiguous(), right=True)
                .squeeze(-1)
                .clamp(max=logits.shape[-1] - 1)
            )
            token = torch.where(self.temperature[:, 0] > 0, sampled, token)
        started = healthy & ~self.inside & (token == self.codec.ids["action"])
        ended = healthy & self.inside & (token == self.codec.ids["action_end"])
        finished = healthy & ~self.inside & (token == self.codec.ids["done"])
        self.history[:, step] = token
        self.has_content = torch.where(
            started | ended, False, self.has_content | (healthy & self.inside & self.content[token])
        )
        self.inside = (self.inside | started) & ~ended
        self.counts = self.counts + ended.long()
        self.status = torch.where(active & ~finite, self.NONFINITE, self.status)
        self.status = torch.where(active & finite & ~valid, self.NO_TOKEN, self.status)
        self.status = torch.where(finished, self.DONE, self.status)
        self.status = torch.where(self.active & (step + 1 >= self.budgets), self.EXHAUSTED, self.status)
        return token

    def retain(self, rows):
        indices = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        for name in (
            "limits",
            "budgets",
            "temperature",
            "status",
            "inside",
            "has_content",
            "counts",
            "history",
        ):
            setattr(self, name, getattr(self, name).index_select(0, indices))
        self.generators = [self.generators[i] for i in rows]
