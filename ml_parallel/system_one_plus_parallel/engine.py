"""One-model choose, suggest, and gated challenge with incremental scoring."""

import math
import time

import torch
from pydantic import ValidationError

from .encoding import Codec
from .errors import InvalidRequestError, ProposalOutputError
from .schema import GenerationOptions, ParallelRequest, normalized


def unique_actions(actions):
    result, seen = [], set()
    for text in actions:
        text = " ".join(text.split())
        key = normalized(text)
        if key and key not in seen:
            result.append(text)
            seen.add(key)
    return result


class Engine:
    def __init__(
        self,
        tokenizer,
        model,
        *,
        expansion_threshold=0.5,
        acceptance_threshold=0.8,
        max_state_tokens=384,
        max_length=1024,
        max_new_tokens=128,
        max_packed_tokens=4096,
        max_cache_tokens=4096,
        max_decode_batch_tokens=16384,
        admission_margin=1.3862943611198906,
        branch_bucket_width=None,
        decode_check_interval=None,
    ):
        if not 0 <= expansion_threshold <= 1 or not 0.5 < acceptance_threshold < 1:
            raise ValueError("invalid expansion or acceptance threshold")
        if max_state_tokens < 1 or max_state_tokens >= max_length or max_new_tokens < 1:
            raise ValueError("invalid context or generation limits")
        self.tokenizer, self.model = tokenizer, model.eval()
        self.codec = Codec(tokenizer)
        self.expansion_threshold, self.acceptance_threshold = expansion_threshold, acceptance_threshold
        self.max_state_tokens, self.max_length, self.max_new_tokens = (
            max_state_tokens,
            max_length,
            max_new_tokens,
        )
        if max_packed_tokens < max_length:
            raise ValueError("max_packed_tokens must be at least max_length")
        self.max_packed_tokens = max_packed_tokens
        if (
            min(max_cache_tokens, max_decode_batch_tokens) < 1
            or not math.isfinite(admission_margin)
            or admission_margin <= 0
        ):
            raise ValueError("invalid decoding capacity or admission margin")
        self.max_cache_tokens, self.max_decode_batch_tokens = max_cache_tokens, max_decode_batch_tokens
        self.admission_margin = admission_margin
        if decode_check_interval is not None:
            if not 1 <= decode_check_interval <= 32:
                raise ValueError("decode_check_interval must be in 1..32")
            self.model.decode_check_interval = decode_check_interval
        if branch_bucket_width is not None:
            if not 0 <= branch_bucket_width <= 256:
                raise ValueError("branch_bucket_width must be in 0..256")
            self.model.branch_bucket_width = branch_bucket_width

    @classmethod
    def from_checkpoint(cls, reference, device="cpu", *, revision=None, **overrides):
        from .model import load_checkpoint

        runtime = {
            name: overrides.pop(name) for name in ("attention_mode", "compute_precision") if name in overrides
        }
        tokenizer, model, metadata = load_checkpoint(reference, device, revision=revision, **runtime)
        settings = {
            key: metadata[key]
            for key in (
                "expansion_threshold",
                "acceptance_threshold",
                "max_state_tokens",
                "max_length",
                "max_packed_tokens",
                "max_cache_tokens",
                "max_decode_batch_tokens",
                "admission_margin",
            )
        }
        return cls(tokenizer, model, **{**settings, **overrides})

    @staticmethod
    def _validate_choices(choices):
        if len(choices) > 255:
            raise InvalidRequestError("at most 255 supplied choices")
        if any(not key.strip() or not value.strip() for key, value in choices.items()):
            raise InvalidRequestError("choice IDs and text must be nonempty")
        if len({normalized(v) for v in choices.values()}) != len(choices):
            raise InvalidRequestError("supplied choices contain duplicate text")

    def _now(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        elif self.model.device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    def _assess(self, state, question, choices):
        self._validate_choices(choices)
        try:
            encoding = self.codec.encode(
                state,
                question,
                list(choices.values()),
                max_state=self.max_state_tokens,
                max_length=self.max_length,
                max_work_tokens=self.max_packed_tokens,
                max_cache_tokens=self.max_cache_tokens,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        return self.model.assess(encoding)

    def _limits(self, choices, n_generate, top_k):
        if not 1 <= n_generate <= 32 or len(choices) + n_generate > 255:
            raise InvalidRequestError("invalid candidate generation budget")
        if top_k is not None and not 1 <= top_k <= 255:
            raise InvalidRequestError("top_k must be in 1..255")

    def _response(
        self,
        entries,
        *,
        top_k=None,
        assessment=None,
        status="not_requested",
        proposals=None,
        error=None,
        latency=None,
        search=None,
        admission_calibrated=False,
    ):
        if any(not math.isfinite(entry["utility"]) for entry in entries):
            raise RuntimeError("non-finite utility")
        values = torch.tensor([e["utility"] for e in entries], dtype=torch.float64)
        probabilities = torch.softmax(values / float(self.model.rank_temperature), 0).tolist()
        ranked = [{**entry, "probability": p} for entry, p in zip(entries, probabilities, strict=True)]
        ranked.sort(key=lambda e: (-e["utility"], normalized(e["text"])))
        winners = [e for e in ranked if abs(e["utility"] - ranked[0]["utility"]) <= 1e-6] if ranked else []
        visible = ranked[:top_k]
        return {
            "choices": visible,
            "candidate_count": len(ranked),
            "selected_choice_id": winners[0]["id"] if len(winners) == 1 else None,
            "abstained": not ranked,
            "omitted_probability_mass": sum(e["probability"] for e in ranked[len(visible) :]),
            "expansion": {
                "probability": assessment.probability if assessment is not None else None,
                "threshold": self.expansion_threshold if assessment is not None else None,
                "acceptance_threshold": self.acceptance_threshold,
                "admission_calibrated": admission_calibrated,
                "admission_calibration_fitted": bool(self.model.admission_calibrated),
                "admission_rule": "calibrated_probability" if admission_calibrated else "utility_margin",
                "admission_margin": self.admission_margin,
                "attempted": status not in {"not_requested", "gate_closed"},
                "status": status,
                "error": error,
            },
            "proposals": proposals or [],
            "search": search,
            "latency_ms": latency or {},
        }

    @staticmethod
    def _entries(choices, values):
        return [
            {"id": key, "text": text, "source": "user", "utility": float(value)}
            for (key, text), value in zip(choices.items(), values, strict=True)
        ]

    def choose(self, state, question, choices, top_k=None):
        if not choices:
            raise InvalidRequestError("choose requires a supplied menu")
        if top_k is not None and not 1 <= top_k <= 255:
            raise InvalidRequestError("invalid top_k")
        start = self._now()
        assessment = self._assess(state, question, choices)
        entries = self._entries(choices, assessment.utilities)
        return self._response(entries, top_k=top_k, latency={"total": 1000 * (self._now() - start)})

    @staticmethod
    def _options(n_generate, proposal_branches=1, temperature=0.0, seed=None):
        try:
            return GenerationOptions(
                n_generate=n_generate, proposal_branches=proposal_branches, temperature=temperature, seed=seed
            ).model_dump()
        except ValidationError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def _generation(self, assessment, n_generate, options=None, result=None):
        if isinstance(result, ProposalOutputError):
            raise result
        if result is not None:
            return result
        return self.model.propose(
            self.codec,
            assessment,
            **(options or self._options(n_generate)),
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            max_cache_tokens=self.max_cache_tokens,
            max_decode_batch_tokens=self.max_decode_batch_tokens,
        )

    def suggest(
        self, state, question, n_generate=3, top_k=None, *, proposal_branches=1, temperature=0.0, seed=None
    ):
        self._limits({}, n_generate, top_k)
        options = self._options(n_generate, proposal_branches, temperature, seed)
        start = self._now()
        assessment = self._assess(state, question, {})
        return self._suggest_assessed(assessment, n_generate, top_k, start, options=options)

    def _suggest_assessed(self, assessment, n_generate, top_k, start, *, options=None, generation=None):
        result = self._generation(assessment, n_generate, options, generation)
        actions = result["actions"]
        actions = unique_actions(actions)
        try:
            values = self.model.score_new(
                assessment, [self.codec.candidate(a) for a in actions], max_length=self.max_length
            )
        except ValueError as exc:
            raise ProposalOutputError(str(exc)) from exc
        entries = [
            {"id": f"generated_{i}", "text": action, "source": "generated", "utility": float(value)}
            for i, (action, value) in enumerate(zip(actions, values, strict=True), 1)
        ]
        return self._response(
            entries,
            top_k=top_k,
            status="suggested" if entries else "empty_proposal",
            search=result.get("search"),
            latency={"total": 1000 * (self._now() - start)},
        )

    def challenge(
        self,
        state,
        question,
        choices,
        n_generate=3,
        top_k=None,
        *,
        proposal_branches=1,
        temperature=0.0,
        seed=None,
    ):
        if not choices:
            raise InvalidRequestError("challenge requires a supplied menu; use suggest for an empty menu")
        self._limits(choices, n_generate, top_k)
        options = self._options(n_generate, proposal_branches, temperature, seed)
        start = self._now()
        assessment = self._assess(state, question, choices)
        return self._challenge_assessed(assessment, choices, n_generate, top_k, start, options=options)

    def _challenge_assessed(
        self, assessment, choices, n_generate, top_k, start, *, options=None, generation=None
    ):
        entries = self._entries(choices, assessment.utilities)
        assessed = self._now()
        if assessment.probability < self.expansion_threshold:
            return self._response(
                entries,
                top_k=top_k,
                assessment=assessment,
                status="gate_closed",
                latency={
                    "assessment": 1000 * (assessed - start),
                    "generation": 0.0,
                    "new_scoring": 0.0,
                    "total": 1000 * (assessed - start),
                },
            )
        try:
            result = self._generation(assessment, n_generate, options, generation)
            generated_at = self._now()
            actions = unique_actions(result["actions"])
            existing = {normalized(v) for v in choices.values()}
            novel = [a for a in actions if normalized(a) not in existing]
            values = self.model.score_new(
                assessment, [self.codec.candidate(a) for a in novel], max_length=self.max_length
            )
            if not torch.isfinite(values).all():
                raise ProposalOutputError("non-finite generated utility")
        except (ProposalOutputError, ValueError) as exc:
            return self._response(
                entries,
                top_k=top_k,
                assessment=assessment,
                status=getattr(exc, "status", "generation_failed"),
                error=str(exc),
                latency={"total": 1000 * (self._now() - start)},
            )
        incumbent = max(entry["utility"] for entry in entries)
        proposals = [
            {"text": a, "admitted": False, "reason": "duplicate"}
            for a in actions
            if normalized(a) in existing
        ]
        from .admission import policy_for

        calibrated = bool(self.model.admission_calibrated) and self.model.admission_policy == policy_for(
            self, options or self._options(n_generate)
        )
        admitted = 0
        for index, (action, value) in enumerate(zip(novel, values, strict=True), 1):
            gap = float(value) - incumbent
            preference = (
                float(torch.sigmoid(self.model.admission_scale * gap + self.model.admission_bias))
                if calibrated
                else None
            )
            accept = gap > 0 and (
                preference >= self.acceptance_threshold if calibrated else gap >= self.admission_margin
            )
            proposals.append(
                {
                    "text": action,
                    "utility": float(value),
                    "preference": preference,
                    "admission_probability": preference,
                    "utility_gap": gap,
                    "admitted": accept,
                    "reason": "improvement" if accept else "below_margin",
                }
            )
            if accept:
                identifier = f"generated_{index}"
                while identifier in choices:
                    identifier = "_" + identifier
                entries.append(
                    {"id": identifier, "text": action, "source": "generated", "utility": float(value)}
                )
                admitted += 1
        finish = self._now()
        status = "repaired" if admitted else ("no_admitted_improvement" if actions else "empty_proposal")
        return self._response(
            entries,
            top_k=top_k,
            assessment=assessment,
            status=status,
            proposals=proposals,
            search=result.get("search"),
            admission_calibrated=calibrated,
            latency={
                "assessment": 1000 * (assessed - start),
                "generation": 1000 * (generated_at - assessed),
                "new_scoring": 1000 * (finish - generated_at),
                "total": 1000 * (finish - start),
            },
        )

    def answer(self, state, questions):
        """Share one state through batched continuations; generation stays question-local."""
        try:
            request = ParallelRequest.model_validate({"state": state, "questions": questions})
        except ValidationError as exc:
            raise InvalidRequestError(str(exc)) from exc
        specs = {key: question.model_dump() for key, question in request.questions.items()}
        for item in specs.values():
            if item["mode"] != "choose":
                self._limits(item["choices"], item["n_generate"], item["top_k"])
        try:
            encoding = self.codec.encode_questions(
                request.state,
                specs,
                max_state=self.max_state_tokens,
                max_length=self.max_length,
                max_packed_tokens=self.max_packed_tokens,
                max_cache_tokens=self.max_cache_tokens,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        start = self._now()
        packed = self.model.assess_questions(encoding)
        assessed = self._now()
        triggered = [
            key
            for key, item in specs.items()
            if item["mode"] == "suggest"
            or (item["mode"] == "challenge" and packed.questions[key].probability >= self.expansion_threshold)
        ]
        generated = (
            self.model.propose_many(
                self.codec,
                [packed.questions[key] for key in triggered],
                question_ids=triggered,
                options=[
                    {name: specs[key][name] for name in GenerationOptions.model_fields} for key in triggered
                ],
                max_new_tokens=self.max_new_tokens,
                max_length=self.max_length,
                max_cache_tokens=self.max_cache_tokens,
                max_decode_batch_tokens=self.max_decode_batch_tokens,
            )
            if triggered
            else []
        )
        generation_results = dict(zip(triggered, generated, strict=True))
        generated_at = self._now()
        answers = {}
        for key, item in specs.items():
            assessment = packed.questions[key]
            question_start = self._now()
            if item["mode"] == "choose":
                answers[key] = self._response(
                    self._entries(item["choices"], assessment.utilities),
                    top_k=item["top_k"],
                )
            elif item["mode"] == "challenge":
                if assessment.probability >= self.expansion_threshold:
                    assessment = packed.branch(key)
                answers[key] = self._challenge_assessed(
                    assessment,
                    item["choices"],
                    item["n_generate"],
                    item["top_k"],
                    question_start,
                    generation=generation_results.get(key),
                    options={name: item[name] for name in GenerationOptions.model_fields},
                )
            else:
                try:
                    answers[key] = self._suggest_assessed(
                        packed.branch(key),
                        item["n_generate"],
                        item["top_k"],
                        question_start,
                        generation=generation_results.get(key),
                    )
                except ProposalOutputError as exc:
                    answers[key] = self._response(
                        [],
                        top_k=item["top_k"],
                        status=exc.status,
                        error=str(exc),
                    )
        for answer in answers.values():
            answer["latency_ms"] = {}  # Timings belong to the shared execution, not individual questions.
        return {
            "answers": answers,
            "usage": {"prefill_tokens": encoding.size, "state_tokens": len(encoding.state)},
            "latency_ms": {
                "assessment": 1000 * (assessed - start),
                "generation": 1000 * (generated_at - assessed),
                "total": 1000 * (self._now() - start),
            },
        }

    def diagnose_ordered_menu(self, state, question, choices):
        a = self._assess(state, question, choices)
        p = torch.softmax(a.utilities / self.model.rank_temperature, 0).tolist()
        return {
            "probabilities": dict(zip(choices, p, strict=True)),
            "utilities": dict(zip(choices, a.utilities.tolist(), strict=True)),
            "expansion_probability": a.probability,
        }

    def propose_for_review(
        self, state, question, choices, n_generate=3, *, proposal_branches=1, temperature=0.0, seed=None
    ):
        self._limits(choices, n_generate, None)
        options = self._options(n_generate, proposal_branches, temperature, seed)
        a = self._assess(state, question, choices)
        try:
            result = self._generation(a, n_generate, options)
            return {
                "expansion_probability": a.probability,
                "raw_output": result["raw_output"],
                "candidates": unique_actions(result["actions"]),
                "error": None,
            }
        except ProposalOutputError as exc:
            return {
                "expansion_probability": a.probability,
                "raw_output": "",
                "candidates": [],
                "error": str(exc),
            }
