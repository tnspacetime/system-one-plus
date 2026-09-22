"""Collect and review actual outputs; accepted means best-quality under the scenario rubric."""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import device_for
from .data import load_split, validate_suite
from .engine import Engine
from .model import load_checkpoint
from .schema import GenerationOptions, Scenario, normalized


class CandidateReview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    label: Literal["accepted", "rejected", "duplicate"] | None = None
    note: str = ""


class ProposalReview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    condition: Literal["incomplete", "complete"]
    checkpoint: str = Field(min_length=1)
    visible_choices: list[str] = Field(min_length=1)
    expansion_probability: float = Field(ge=0, le=1)
    raw_output: str
    candidates: list[CandidateReview]
    generation_error: str | None = None
    generation_options: GenerationOptions | None = None

    @model_validator(mode="after")
    def unique_candidates(self):
        keys = [normalized(candidate.text) for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate text must be unique within a review record")
        return self


def read_reviews(path: Path) -> list[ProposalReview]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(ProposalReview.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"empty proposal review file: {path}")
    return records


def load_reviewed_proposals(path: Path, scenarios: list[Scenario]):
    """Require completed, non-conflicting human labels and group them by scenario."""
    by_id = {scenario.id: scenario for scenario in scenarios}
    scenario_ids = set(by_id)
    grouped = {scenario.id: {"accepted": [], "rejected": []} for scenario in scenarios}
    labels = {}
    record_ids = set()
    reviewed_candidates = 0
    generation_failures = 0
    for record in read_reviews(path):
        if record.id in record_ids:
            raise ValueError(f"duplicate proposal review id: {record.id}")
        record_ids.add(record.id)
        if record.scenario_id not in scenario_ids:
            raise ValueError(
                f"review references a scenario outside this training split: {record.scenario_id}"
            )
        if record.generation_error:
            generation_failures += 1
        scenario = by_id[record.scenario_id]
        expected = (
            scenario.complete_choices if record.condition == "complete" else scenario.incomplete_choices
        )
        visible = {normalized(value) for value in record.visible_choices}
        if visible != {normalized(value) for value in expected}:
            raise ValueError("review menu does not match its frozen scenario condition")
        known_good = {
            normalized(scenario.best_choice),
            *(normalized(v) for v in scenario.acceptable_equivalents),
        }
        known_bad = {normalized(v) for v in scenario.complete_choices if v != scenario.best_choice}
        known_bad.update(normalized(v) for v in scenario.rejected_proposals)
        for candidate in record.candidates:
            if candidate.label is None:
                raise ValueError(f"pending candidate review: {record.id}: {candidate.text}")
            key = normalized(candidate.text)
            if key in visible and candidate.label != "duplicate":
                raise ValueError(
                    f"candidate already visible and must be labeled duplicate: {record.id}: {candidate.text}"
                )
            if candidate.label == "duplicate":
                continue
            if (candidate.label == "rejected" and key in known_good) or (
                candidate.label == "accepted" and key in known_bad
            ):
                raise ValueError("review contradicts frozen action-quality labels")
            reviewed_candidates += 1
            previous = labels.get((record.scenario_id, key))
            if previous is not None and previous != candidate.label:
                raise ValueError(f"conflicting reviews for {record.scenario_id}: {candidate.text}")
            labels[(record.scenario_id, key)] = candidate.label
            bucket = grouped[record.scenario_id][candidate.label]
            if key not in {normalized(value) for value in bucket}:
                bucket.append(candidate.text)
    return grouped, {
        "records": len(record_ids),
        "reviewed_candidates": reviewed_candidates,
        "accepted": sum(len(value["accepted"]) for value in grouped.values()),
        "rejected": sum(len(value["rejected"]) for value in grouped.values()),
        "generation_failures": generation_failures,
    }


def collect(
    engine: Engine,
    rows: list[Scenario],
    checkpoint: str,
    n_generate: int,
    *,
    proposal_branches=1,
    temperature=0.0,
    seed=None,
):
    options = GenerationOptions(
        n_generate=n_generate, proposal_branches=proposal_branches, temperature=temperature, seed=seed
    )
    records = []
    for row in rows:
        for condition, choices in (
            ("incomplete", row.incomplete_choices),
            ("complete", row.complete_choices),
        ):
            result = engine.propose_for_review(
                row.state,
                row.question,
                {f"c{index}": value for index, value in enumerate(choices)},
                **options.model_dump(),
            )
            records.append(
                ProposalReview(
                    id=f"{row.id}/{condition}",
                    scenario_id=row.id,
                    condition=condition,
                    checkpoint=checkpoint,
                    visible_choices=list(choices),
                    expansion_probability=result["expansion_probability"],
                    raw_output=result["raw_output"],
                    candidates=[CandidateReview(text=text) for text in result["candidates"]],
                    generation_error=result["error"],
                    generation_options=options,
                )
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--checkpoint", required=True)
    collect_parser.add_argument("--checkpoint-revision")
    collect_parser.add_argument("--suite", type=Path, required=True)
    collect_parser.add_argument("--split", choices=["train", "development", "test"], default="train")
    collect_parser.add_argument("--allow-test", action="store_true")
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    collect_parser.add_argument("--n-generate", type=int, default=8, choices=range(1, 33))
    collect_parser.add_argument("--proposal-branches", type=int, default=1)
    collect_parser.add_argument("--temperature", type=float, default=0.0)
    collect_parser.add_argument("--seed", type=int)
    collect_parser.add_argument("--max-new-tokens", type=int, default=128)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("reviews", type=Path)
    validate_parser.add_argument("--suite", type=Path, required=True)
    validate_parser.add_argument("--split", choices=["train", "development", "test"], default="train")
    validate_parser.add_argument("--allow-test", action="store_true")

    args = parser.parse_args()
    validate_suite(args.suite)
    rows = load_split(args.suite, args.split, allow_test=args.allow_test)
    if args.command == "validate":
        _, stats = load_reviewed_proposals(args.reviews, rows)
        print(json.dumps(stats, indent=2))
        return
    if args.out.exists():
        raise FileExistsError("refusing to overwrite proposal collection")
    tokenizer, model, metadata = load_checkpoint(
        args.checkpoint,
        device_for(args.device),
        revision=args.checkpoint_revision,
    )
    records = collect(
        Engine(
            tokenizer,
            model,
            expansion_threshold=metadata["expansion_threshold"],
            acceptance_threshold=metadata["acceptance_threshold"],
            max_state_tokens=metadata["max_state_tokens"],
            max_length=metadata["max_length"],
            max_packed_tokens=metadata["max_packed_tokens"],
            max_cache_tokens=metadata["max_cache_tokens"],
            max_decode_batch_tokens=metadata["max_decode_batch_tokens"],
            admission_margin=metadata["admission_margin"],
            max_new_tokens=args.max_new_tokens,
        ),
        rows,
        args.checkpoint,
        args.n_generate,
        proposal_branches=args.proposal_branches,
        temperature=args.temperature,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(record.model_dump_json() + "\n" for record in records))
    print(
        json.dumps(
            {
                "records": len(records),
                "candidates": sum(len(record.candidates) for record in records),
                "generation_failures": sum(bool(record.generation_error) for record in records),
                "output": str(args.out),
                "next": "label every candidate accepted, rejected, or duplicate; then run sop-parallel-harvest validate",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
