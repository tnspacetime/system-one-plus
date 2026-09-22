"""Shared Qwen/LoRA with independent utility, set omission, and vocabulary readouts."""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import compute_dtype, use_flash
from .decoding import DecodeCache
from .encoding import ENCODING_VERSION, TOKENS, Codec, Encoding, ParallelEncoding
from .errors import ProposalOutputError
from .proposal import ActionGrammar, BatchGrammar
from .schema import GenerationOptions, normalized
from .staged import ATTENTION_BACKEND, EXECUTION_VERSION, BranchCache, Continuation, run_continuations

ARCHITECTURE = "parallel_utility_set_attention_v1"
CHECKPOINT_VERSION = 4
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Assessment:
    encoding: Encoding
    utilities: torch.Tensor
    omission_logit: torch.Tensor | None
    probability: float | None
    cache: BranchCache


@dataclass
class ParallelAssessment:
    encoding: ParallelEncoding
    questions: dict[str, Assessment]

    def branch(self, key):
        """Branches already own their suffixes and share immutable prefix segments."""
        return self.questions[key]


class SystemOnePlusModel(nn.Module):
    def __init__(
        self,
        base_model,
        *,
        revision=None,
        lora_rank=16,
        head_dim=128,
        omission_heads=4,
        omission_layers=2,
        branch_batch_size=16,
        branch_bucket_width=0,
        device="cpu",
        attention_mode="auto",
        compute_precision="auto",
        decode_check_interval=8,
    ):
        super().__init__()
        if head_dim < 1 or omission_heads < 1 or head_dim % omission_heads or omission_layers < 1:
            raise ValueError(
                "omission attention requires positive dimensions and head_dim divisible by omission_heads"
            )
        if branch_batch_size < 1 or branch_bucket_width < 0:
            raise ValueError("branch_batch_size must be positive")
        if not 1 <= decode_check_interval <= 32:
            raise ValueError("decode_check_interval must be in 1..32")
        self.attention_mode, self.compute_precision = attention_mode, compute_precision
        self.decode_check_interval = decode_check_interval
        dtype = compute_dtype(device, compute_precision, attention_mode)
        self._runtime_key = (torch.device(device), compute_precision, attention_mode)
        self._runtime_dtype = dtype
        self.branch_batch_size = branch_batch_size
        self.branch_bucket_width = branch_bucket_width
        self.gradient_checkpointing = False
        self.admission_policy = None
        self.base_model, self.base_revision = str(base_model), revision
        self.lora_rank, self.head_dim = lora_rank, head_dim
        self.omission_heads, self.omission_layers = omission_heads, omission_layers
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model, revision=revision, torch_dtype=dtype, attn_implementation=ATTENTION_BACKEND
        )
        if self.lm.config.model_type != "qwen2":
            raise ValueError("this encoding is verified for Qwen2/Qwen2.5 models")
        if self.lm.config.use_sliding_window or getattr(self.lm.config, "attention_dropout", 0.0):
            raise ValueError("staged execution requires full attention and zero attention dropout")
        self.lm.config.sop_attention_mode = attention_mode
        self.lm.requires_grad_(False)
        self.lm.model = get_peft_model(
            self.lm.model,
            LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.0,
                target_modules=LORA_TARGETS,
            ),
        )
        hidden = self.lm.config.hidden_size
        self.utility = nn.Linear(hidden, 1)
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden + 1, head_dim), nn.SiLU(), nn.Linear(head_dim, head_dim)
        )
        self.context_projection = nn.Linear(hidden, head_dim)
        # Each layer is independently initialized. No positions or causal mask are
        # added: candidate permutations preserve the context token's readout.
        self.set_attention = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=head_dim,
                    nhead=omission_heads,
                    dim_feedforward=4 * head_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(omission_layers)
            ]
        )
        self.omission = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 1)
        )
        self.register_buffer("omission_scale", torch.tensor(1.0))
        self.register_buffer("omission_bias", torch.tensor(0.0))
        self.register_buffer("rank_temperature", torch.tensor(1.0))
        self.register_buffer("admission_scale", torch.tensor(1.0))
        self.register_buffer("admission_bias", torch.tensor(0.0))
        self.register_buffer("admission_calibrated", torch.tensor(False))
        self.to(device)

    @property
    def device(self):
        return self.utility.weight.device

    @property
    def runtime(self):
        return {
            "attention": "flash_attention_2"
            if use_flash(self.device, self.attention_mode)
            else "reference_sdpa",
            "compute_precision": str(self._runtime_dtype),
        }

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _heads(self, hidden, ends, prefix_length):
        hidden = hidden.to(self.utility.weight.dtype)
        indices = torch.tensor(ends, dtype=torch.long, device=self.device)
        actions = hidden.index_select(0, indices)
        utilities = self.utility(actions).squeeze(-1).float()
        if not ends:
            return utilities, None
        elements = self.candidate_projection(
            torch.cat([actions, utilities.detach()[:, None].to(actions.dtype)], -1)
        )
        context = self.context_projection(hidden[prefix_length - 1]).unsqueeze(0)
        tokens = torch.cat([context, elements], dim=0).unsqueeze(0)
        for layer in self.set_attention:
            tokens = layer(tokens)
        return utilities, self.omission(tokens[0, 0]).squeeze(-1).float()

    def _compute(self):
        key = (self.device, self.compute_precision, self.attention_mode)
        if key != self._runtime_key:
            self._runtime_dtype = compute_dtype(*key)
            self._runtime_key = key
        dtype = self._runtime_dtype
        return (
            torch.autocast("cuda", dtype=dtype)
            if self.device.type == "cuda" and dtype != torch.float32
            else contextlib.nullcontext()
        )

    def _project(self, hidden):
        with self._compute():
            return self.lm.lm_head(hidden).float()

    def _continue(self, branches):
        with self._compute():
            return run_continuations(
                self.lm.model,
                branches,
                device=self.device,
                checkpointing=self.gradient_checkpointing and self.training,
                max_batch_size=self.branch_batch_size,
                bucket_width=self.branch_bucket_width,
            )

    def forward(self, encoding: Encoding | ParallelEncoding, *, use_cache=False):
        if isinstance(encoding, ParallelEncoding):
            return self.forward_questions(encoding, use_cache=use_cache)
        if not 0 < encoding.state_length < len(encoding.prefix):
            raise ValueError("invalid state/question boundary")
        parallel = ParallelEncoding(encoding.prefix[: encoding.state_length], {"question": encoding})
        output = self.forward_questions(parallel, use_cache=use_cache)
        return output["questions"]["question"]

    def forward_questions(self, encoding, *, use_cache=False):
        for question in encoding.questions.values():
            if question.next_position > self.lm.config.max_position_embeddings:
                raise ValueError("question exceeds model position context")
        state_length = len(encoding.state)
        state = self._continue([Continuation(encoding.state, tuple(range(state_length)))])[0]
        items = list(encoding.questions.items())
        contexts = self._continue(
            [
                Continuation(q.prefix[state_length:], tuple(range(state_length, len(q.prefix))), state.cache)
                for _, q in items
            ]
        )
        candidates, owners = [], []
        for index, (_, question) in enumerate(items):
            for candidate in question.candidates:
                candidates.append(
                    Continuation(
                        candidate,
                        tuple(range(len(question.prefix), len(question.prefix) + len(candidate))),
                        contexts[index].cache,
                    )
                )
                owners.append(index)
        results = self._continue(candidates)
        by_question = [[] for _ in items]
        for owner, result in zip(owners, results, strict=True):
            by_question[owner].append(result)
        outputs, proposals, proposal_keys = {}, [], []
        for index, (key, question) in enumerate(items):
            context = contexts[index]
            actions = by_question[index]
            hidden = torch.stack([context.hidden[-1], *(a.hidden[-1] for a in actions)])
            utilities, omission = self._heads(hidden, list(range(1, len(actions) + 1)), 1)
            # Menu assembly joins KV suffix references; the state/question occurs once.
            menu = context.cache
            for action in actions:
                menu = menu.append(action.cache.segments[-1])
            outputs[key] = {
                "utilities": utilities,
                "omission_logit": omission,
                "proposal_logits": None,
                "cache": menu if use_cache else None,
            }
            if question.proposal:
                start = question.next_position - len(question.proposal)
                proposals.append(
                    Continuation(question.proposal, tuple(range(start, question.next_position)), menu)
                )
                proposal_keys.append(key)
        for key, result in zip(proposal_keys, self._continue(proposals), strict=True):
            outputs[key]["proposal_logits"] = self._project(result.hidden)
            if use_cache:
                outputs[key]["cache"] = result.cache
        return {"questions": outputs}

    def losses(self, encoding, *, preferences=(), omission_target=None):
        output = self(encoding)
        return self._losses(encoding, output, preferences=preferences, omission_target=omission_target)

    def losses_questions(self, encoding, supervision):
        if set(supervision) != set(encoding.questions):
            raise ValueError("supervision must match the encoded question IDs")
        output = self(encoding)
        return {
            key: self._losses(question, output["questions"][key], **supervision[key])
            for key, question in encoding.questions.items()
        }

    @staticmethod
    def _losses(encoding, output, *, preferences=(), omission_target=None):
        u, losses = output["utilities"], {}
        if preferences:
            pairs = torch.tensor(preferences, dtype=torch.long, device=u.device)
            losses["score"] = F.softplus(-(u[pairs[:, 0]] - u[pairs[:, 1]])).mean()
        if omission_target is not None:
            if output["omission_logit"] is None:
                raise ValueError("empty menus cannot have omission targets")
            losses["omission"] = F.binary_cross_entropy_with_logits(
                output["omission_logit"], u.new_tensor(float(omission_target))
            )
        if len(encoding.proposal) > 1:
            targets = torch.tensor(encoding.proposal[1:], device=u.device)
            losses["generation"] = F.cross_entropy(output["proposal_logits"][:-1], targets)
        if not losses:
            raise ValueError("example has no supervised objective")
        return losses

    def missing_probability(self, logit):
        return torch.sigmoid(self.omission_scale * logit + self.omission_bias)

    @torch.no_grad()
    def assess(self, encoding):
        if self.training:
            raise RuntimeError("inference requires model.eval()")
        if encoding.proposal:
            raise ValueError("assessment must not include proposal targets")
        result = self(encoding, use_cache=True)
        z = result["omission_logit"]
        if not torch.isfinite(result["utilities"]).all() or (z is not None and not torch.isfinite(z)):
            raise RuntimeError("non-finite assessment output")
        return Assessment(
            encoding,
            result["utilities"],
            z,
            float(self.missing_probability(z)) if z is not None else None,
            result["cache"],
        )

    @torch.no_grad()
    def assess_questions(self, encoding):
        if self.training:
            raise RuntimeError("inference requires model.eval()")
        if any(q.proposal for q in encoding.questions.values()):
            raise ValueError("assessment must not include proposal targets")
        output = self(encoding, use_cache=True)
        assessments = {}
        for key, result in output["questions"].items():
            z = result["omission_logit"]
            if not torch.isfinite(result["utilities"]).all() or (z is not None and not torch.isfinite(z)):
                raise RuntimeError("non-finite assessment output")
            assessments[key] = Assessment(
                encoding.questions[key],
                result["utilities"],
                z,
                float(self.missing_probability(z)) if z is not None else None,
                result["cache"],
            )
        return ParallelAssessment(encoding, assessments)

    def _cached(self, tokens, positions, cache):
        result = self._continue([Continuation(tuple(tokens), tuple(positions), cache)])[0]
        return result.hidden, result.cache

    @torch.no_grad()
    def score_new(self, assessment, candidates, *, max_length=1024):
        """Batch new candidates against the original state/question prefix only."""
        if self.training:
            raise RuntimeError("inference requires model.eval()")
        prefix_length = len(assessment.encoding.prefix)
        prefix = assessment.cache.prefix(prefix_length)
        branches = []
        for candidate in candidates:
            if prefix_length + len(candidate) > min(max_length, self.lm.config.max_position_embeddings):
                raise ValueError("generated candidate exceeds scoring context")
            branches.append(
                Continuation(
                    tuple(candidate),
                    tuple(range(prefix_length, prefix_length + len(candidate))),
                    prefix,
                )
            )
        outputs = self._continue(branches)
        if not outputs:
            return self.utility.weight.new_empty(0)
        hidden = torch.stack([out.hidden[-1] for out in outputs]).to(self.utility.weight.dtype)
        return self.utility(hidden).squeeze(-1).float()

    @torch.no_grad()
    def propose(
        self,
        codec,
        assessment,
        *,
        n_generate=3,
        max_new_tokens=128,
        max_length=1024,
        max_cache_tokens=4096,
        max_decode_batch_tokens=16384,
        proposal_branches=1,
        temperature=0.0,
        seed=None,
        question_id="question",
    ):
        result = self.propose_many(
            codec,
            [assessment],
            question_ids=[question_id],
            options=[
                dict(
                    n_generate=n_generate,
                    proposal_branches=proposal_branches,
                    temperature=temperature,
                    seed=seed,
                )
            ],
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            max_cache_tokens=max_cache_tokens,
            max_decode_batch_tokens=max_decode_batch_tokens,
        )[0]
        if isinstance(result, ProposalOutputError):
            raise result
        return result

    @torch.no_grad()
    def propose_many(
        self,
        codec,
        assessments,
        *,
        question_ids,
        options=None,
        max_new_tokens=128,
        max_length=1024,
        max_cache_tokens=4096,
        max_decode_batch_tokens=16384,
    ):
        """Decode question/search branches in bounded batches; failures stay row-local."""
        if self.training:
            raise RuntimeError("inference requires model.eval()")
        if min(max_new_tokens, max_cache_tokens, max_decode_batch_tokens) < 1:
            raise ValueError("invalid generation budget")
        if len(question_ids) != len(assessments) or any(
            not isinstance(key, str) or not key.strip() for key in question_ids
        ):
            raise ValueError("one nonempty stable question ID is required per assessment")
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question IDs must be unique")
        options = [
            GenerationOptions.model_validate(o)
            for o in (options if options is not None else [{} for _ in assessments])
        ]
        if len(options) != len(assessments):
            raise ValueError("one set of generation options is required per question")
        completed, errors, rows = [[] for _ in assessments], [[] for _ in assessments], []
        for owner, (question_id, assessment, option) in enumerate(
            zip(question_ids, assessments, options, strict=True)
        ):
            encoding = assessment.encoding
            budget = min(
                max_new_tokens,
                min(max_length, self.lm.config.max_position_embeddings) - encoding.next_position,
                max_cache_tokens - assessment.cache.get_seq_length(),
            )
            capacity = assessment.cache.get_seq_length() + budget
            if budget < 1 or capacity > max_decode_batch_tokens:
                errors[owner].append(ProposalOutputError("no decoding capacity", status="budget_exhausted"))
                continue
            for branch in range(option.proposal_branches):
                count = option.n_generate // option.proposal_branches + (
                    branch < option.n_generate % option.proposal_branches
                )
                seed = option.seed if option.seed is not None else secrets.randbits(63)
                # IDs survive question reordering, gate filtering, and batch splitting.
                # Structured serialization avoids ambiguous string concatenation.
                stream = json.dumps([seed, question_id, branch], separators=(",", ":")).encode()
                seed = int.from_bytes(hashlib.sha256(stream).digest()[:8], "little") % (2**63)
                rows.append(
                    dict(
                        owner=owner,
                        branch=branch,
                        parent=assessment.cache,
                        budget=budget,
                        position=encoding.next_position,
                        count=count,
                        temperature=option.temperature,
                        seed=seed,
                        capacity=capacity,
                    )
                )
        # Group by capacity, bounding both row count and allocated KV token slots.
        batches, batch = [], []
        for row in sorted(rows, key=lambda r: r["capacity"]):
            if batch and (
                len(batch) >= self.branch_batch_size
                or (len(batch) + 1) * row["capacity"] > max_decode_batch_tokens
            ):
                batches.append(batch)
                batch = []
            batch.append(row)
        if batch:
            batches.append(batch)
        for active in batches:
            cache = DecodeCache([r["parent"] for r in active], [r["budget"] for r in active])
            grammar = BatchGrammar(codec, active, self.lm.config.vocab_size, self.device)
            tokens = torch.full((len(active),), codec.ids["propose"], device=self.device, dtype=torch.long)
            positions = torch.tensor([r["position"] for r in active], device=self.device)
            max_steps = max(r["budget"] for r in active)
            for step in range(max_steps):
                with self._compute():
                    hidden = cache.step(
                        self.lm.model, tokens, positions + step, self.device, active=grammar.active
                    )
                    logits = self._project(hidden)
                tokens = grammar.select(logits, step)
                if (step + 1) % self.decode_check_interval and step + 1 < max_steps:
                    continue
                # One small batch status transfer per block, never vocabulary logits
                # or per-row tensor scalars. Completed text is transferred only here.
                statuses = grammar.status.cpu().tolist()
                finished = [i for i, status in enumerate(statuses) if status == BatchGrammar.DONE]
                histories = grammar.history[finished, : step + 1].cpu().tolist() if finished else []
                for index, history in zip(finished, histories, strict=True):
                    row = active[index]
                    parsed = ActionGrammar(codec, row["count"])
                    try:
                        for token in history:
                            parsed.consume(token)
                            if parsed.finished:
                                break
                        completed[row["owner"]].append((row["branch"], parsed))
                    except ProposalOutputError as exc:
                        errors[row["owner"]].append(exc)
                for row, status in zip(active, statuses, strict=True):
                    if status in (BatchGrammar.NONFINITE, BatchGrammar.NO_TOKEN, BatchGrammar.EXHAUSTED):
                        message = {
                            BatchGrammar.NONFINITE: "non-finite proposal logits",
                            BatchGrammar.NO_TOKEN: "grammar has no valid next token",
                            BatchGrammar.EXHAUSTED: "proposal did not terminate within its token budget",
                        }[status]
                        errors[row["owner"]].append(
                            ProposalOutputError(
                                message,
                                status="budget_exhausted"
                                if status == BatchGrammar.EXHAUSTED
                                else "generation_failed",
                            )
                        )
                keep = [i for i, status in enumerate(statuses) if status == BatchGrammar.ACTIVE]
                if not keep:
                    break
                if len(keep) != len(active):
                    cache.retain(keep)
                    grammar.retain(keep)
                    tokens, positions = tokens[keep], positions[keep]
                    active = [active[i] for i in keep]
        results = []
        for owner, option in enumerate(options):
            if not completed[owner]:
                results.append(errors[owner][0])
                continue
            grammars = [g for _, g in sorted(completed[owner], key=lambda pair: pair[0])]
            actions, seen = [], set()
            for grammar in grammars:
                for action in grammar.actions:
                    if normalized(action) not in seen:
                        actions.append(action)
                        seen.add(normalized(action))
            results.append(
                {
                    "actions": actions,
                    "token_ids": [t for g in grammars for t in g.tokens],
                    "raw_output": "\n".join(
                        codec.tokenizer.decode(g.tokens, skip_special_tokens=False) for g in grammars
                    ),
                    "search": {
                        **option.model_dump(),
                        "completed_branches": len(grammars),
                        "errors": [{"status": e.status, "message": str(e)} for e in errors[owner]],
                    },
                }
            )
        return results


def load_tokenizer(name, revision=None):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    Codec(tokenizer)
    return tokenizer


def resolve_checkpoint(reference, revision=None):
    path = Path(reference).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or str(reference).startswith(("./", "../", "~")):
        raise FileNotFoundError(reference)
    return Path(
        snapshot_download(
            str(reference),
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"],
        )
    )


def save_checkpoint(model, tokenizer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.lm.model.save_pretrained(directory / "adapter")
    tokenizer.save_pretrained(directory / "tokenizer")
    heads = {name: value for name, value in model.state_dict().items() if not name.startswith("lm.")}
    torch.save(heads, directory / "heads.pt")
    metadata = {
        **metadata,
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "encoding_version": ENCODING_VERSION,
        "base_model": model.base_model,
        "base_revision": model.base_revision,
        "lora_rank": model.lora_rank,
        "head_dim": model.head_dim,
        "omission_heads": model.omission_heads,
        "omission_layers": model.omission_layers,
        "token_ids": Codec(tokenizer).ids,
        "token_spellings": TOKENS,
        "attention_backend": ATTENTION_BACKEND,
        "execution_version": EXECUTION_VERSION,
        "branch_batch_size": model.branch_batch_size,
        "branch_bucket_width": model.branch_bucket_width,
        "attention_mode": model.attention_mode,
        "compute_precision": model.compute_precision,
        "decode_check_interval": model.decode_check_interval,
        "max_cache_tokens": metadata.get("max_cache_tokens", 4096),
        "max_decode_batch_tokens": metadata.get("max_decode_batch_tokens", 16384),
        "admission_margin": metadata.get("admission_margin", 1.3862943611198906),
        "max_packed_tokens": metadata.get("max_packed_tokens", 4096),
    }
    (directory / "system_one_plus_parallel.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


def load_checkpoint(
    reference, device="cpu", *, revision=None, trainable=False, attention_mode=None, compute_precision=None
):
    directory = resolve_checkpoint(reference, revision)
    metadata = json.loads((directory / "system_one_plus_parallel.json").read_text())
    legacy_version = metadata.get("checkpoint_version")
    if metadata.get("architecture") == ARCHITECTURE and metadata.get("encoding_version") == ENCODING_VERSION:
        previous = {
            1: ("eager", None),
            2: (ATTENTION_BACKEND, "staged_sdpa_v1"),
            3: (ATTENTION_BACKEND, "staged_sdpa_v2"),
        }
        if legacy_version in previous:
            backend, execution = previous[legacy_version]
            if metadata.get("attention_backend") == backend and (
                execution is None or metadata.get("execution_version") == execution
            ):
                metadata = {
                    **metadata,
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "attention_backend": ATTENTION_BACKEND,
                    "execution_version": EXECUTION_VERSION,
                    "attention_mode": "auto",
                    "compute_precision": "auto",
                    "decode_check_interval": 8,
                    "loaded_from_execution": {
                        "checkpoint_version": legacy_version,
                        "attention_backend": backend,
                        "execution_version": execution,
                    },
                }
                if legacy_version in (1, 2):
                    metadata.update(
                        branch_batch_size=metadata.get("branch_batch_size", 16),
                        branch_bucket_width=0,
                        max_cache_tokens=metadata.get("max_length", 1024),
                        max_decode_batch_tokens=16384,
                        admission_margin=1.3862943611198906,
                    )
                metadata["admission_calibration"] = {"fitted": False, "invalidated_by": "proposal_decoder_v2"}
    required = {
        "base_model",
        "base_revision",
        "lora_rank",
        "head_dim",
        "omission_heads",
        "omission_layers",
        "token_ids",
        "token_spellings",
        "attention_backend",
        "execution_version",
        "branch_batch_size",
        "branch_bucket_width",
        "attention_mode",
        "compute_precision",
        "decode_check_interval",
        "max_cache_tokens",
        "max_decode_batch_tokens",
        "admission_margin",
        "expansion_threshold",
        "acceptance_threshold",
        "max_state_tokens",
        "max_length",
        "max_packed_tokens",
    }
    if not required <= metadata.keys():
        raise ValueError("incomplete utility checkpoint metadata")
    if (
        metadata.get("architecture"),
        metadata.get("checkpoint_version"),
        metadata.get("encoding_version"),
    ) != (ARCHITECTURE, CHECKPOINT_VERSION, ENCODING_VERSION):
        raise ValueError("incompatible checkpoint: expected parallel utility with set attention")
    if (
        metadata["token_spellings"] != TOKENS
        or metadata["attention_backend"] != ATTENTION_BACKEND
        or metadata["execution_version"] != EXECUTION_VERSION
    ):
        raise ValueError("unsupported encoding or attention backend")
    tokenizer = load_tokenizer(directory / "tokenizer")
    if Codec(tokenizer).ids != metadata["token_ids"]:
        raise ValueError("checkpoint tokenizer/control IDs do not match")
    model = SystemOnePlusModel(
        metadata["base_model"],
        revision=metadata["base_revision"],
        lora_rank=metadata["lora_rank"],
        head_dim=metadata["head_dim"],
        omission_heads=metadata["omission_heads"],
        omission_layers=metadata["omission_layers"],
        branch_batch_size=metadata["branch_batch_size"],
        branch_bucket_width=metadata["branch_bucket_width"],
        device=device,
        attention_mode=attention_mode or metadata["attention_mode"],
        compute_precision=compute_precision or metadata["compute_precision"],
        decode_check_interval=metadata["decode_check_interval"],
    )
    adapter = load_file(str(directory / "adapter/adapter_model.safetensors"))
    result = set_peft_model_state_dict(model.lm.model, adapter)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("incompatible adapter state")
    heads = torch.load(directory / "heads.pt", map_location=device, weights_only=True)
    if legacy_version in (1, 2, 3):
        for name in ("admission_scale", "admission_bias", "admission_calibrated"):
            heads[name] = getattr(model, name).clone()
    expected = {name for name in model.state_dict() if not name.startswith("lm.")}
    if set(heads) != expected:
        raise ValueError("incomplete utility head/calibration state")
    model.load_state_dict(heads, strict=False)
    if not all(torch.isfinite(v).all() for v in heads.values()):
        raise ValueError("non-finite checkpoint head state")
    if model.rank_temperature <= 0 or model.omission_scale <= 0 or model.admission_scale <= 0:
        raise ValueError("invalid checkpoint calibration scale")
    model.admission_policy = metadata.get("admission_calibration", {}).get("policy")
    if bool(model.admission_calibrated) and not model.admission_policy:
        raise ValueError("fitted admission calibration requires its generation policy")
    if not trainable:
        model.requires_grad_(False)
    model.train(trainable)
    return tokenizer, model, metadata
