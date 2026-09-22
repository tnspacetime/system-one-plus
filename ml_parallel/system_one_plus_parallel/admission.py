"""Collect and calibrate generated-vs-incumbent judgments on held-out development groups."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import functional as F

from .data import digest, load_split, validate_suite
from .engine import Engine
from .model import resolve_checkpoint, save_checkpoint
from .schema import GenerationOptions, normalized


class AdmissionPair(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    scenario_id: str
    condition: Literal["complete", "incomplete"]
    checkpoint_fingerprint: str
    policy: dict
    incumbent: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    better_than_incumbent: bool | None = None
    note: str = ""


def fingerprint(reference):
    path = resolve_checkpoint(reference)
    files = [
        path / "heads.pt",
        path / "adapter/adapter_model.safetensors",
        path / "system_one_plus_parallel.json",
    ]
    return hashlib.sha256("".join(digest(p) for p in files).encode()).hexdigest()


def policy_for(engine, options):
    return {
        "decoder": "batched_device_grammar_v2",
        **engine.model.runtime,
        **{name: options[name] for name in ("n_generate", "proposal_branches", "temperature")},
        "max_new_tokens": engine.max_new_tokens,
        "expansion_threshold": engine.expansion_threshold,
        "max_length": engine.max_length,
        "max_cache_tokens": engine.max_cache_tokens,
    }


def collect_pairs(engine, scenarios, checkpoint_fingerprint, options):
    records, failures = [], []
    for scenario in scenarios:
        for condition in ("complete", "incomplete"):
            menu = getattr(scenario, f"{condition}_choices")
            choices = {str(i): text for i, text in enumerate(menu)}
            baseline = engine.choose(scenario.state, scenario.question, choices)
            # Ties use the same deterministic text ordering as serving.
            incumbent = baseline["choices"][0]["text"]
            result = engine.challenge(scenario.state, scenario.question, choices, **options)
            if result["expansion"]["error"] or (result.get("search") or {}).get("errors"):
                failures.append(
                    {
                        "scenario_id": scenario.id,
                        "condition": condition,
                        "error": result["expansion"]["error"],
                        "search": result.get("search"),
                    }
                )
            for proposal in result["proposals"]:
                if proposal["reason"] == "duplicate":
                    continue
                records.append(
                    AdmissionPair(
                        scenario_id=scenario.id,
                        condition=condition,
                        checkpoint_fingerprint=checkpoint_fingerprint,
                        policy=policy_for(engine, options),
                        incumbent=incumbent,
                        candidate=proposal["text"],
                    )
                )
    return records, failures


def reviewed_pairs(path, scenarios, engine, expected_fingerprint):
    scenarios = {row.id: row for row in scenarios}
    records = [
        AdmissionPair.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("empty admission review file")
    policy = records[0].policy
    options = GenerationOptions.model_validate(
        {k: policy[k] for k in ("n_generate", "proposal_branches", "temperature")}
    )
    if policy != policy_for(engine, options.model_dump()):
        raise ValueError("review generation policy differs from runtime policy/budgets")
    seen, groups, pairs = set(), set(), []
    for row in records:
        if row.checkpoint_fingerprint != expected_fingerprint or row.policy != policy:
            raise ValueError("reviews must use this exact checkpoint and one generation policy")
        if row.scenario_id not in scenarios:
            raise ValueError("admission fitting only accepts frozen development scenarios")
        if row.better_than_incumbent is None:
            raise ValueError("pending better_than_incumbent review")
        scenario = scenarios[row.scenario_id]
        menu = getattr(scenario, f"{row.condition}_choices")
        key = (row.scenario_id, row.condition, normalized(row.candidate))
        if key in seen:
            raise ValueError("duplicate admission review pair")
        seen.add(key)
        if normalized(row.candidate) in {normalized(c) for c in menu}:
            raise ValueError("admission candidates must be novel to the visible menu")
        assessment = engine._assess(
            scenario.state, scenario.question, {str(i): c for i, c in enumerate(menu)}
        )
        order = sorted(range(len(menu)), key=lambda i: (-float(assessment.utilities[i]), normalized(menu[i])))
        best = order[0]
        if menu[best] != row.incumbent:
            raise ValueError("review incumbent differs from this model's best supplied action")
        if assessment.probability < engine.expansion_threshold:
            raise ValueError("review does not match the runtime expansion gate")
        utility = engine.model.score_new(
            assessment, [engine.codec.candidate(row.candidate)], max_length=engine.max_length
        )[0]
        pairs.append((float(utility - assessment.utilities[best]), int(row.better_than_incumbent)))
        groups.add(scenario.group_id)
    return pairs, groups, policy


def metrics(model, pairs, threshold):
    if not pairs:
        return None
    gaps = torch.tensor([gap for gap, _ in pairs], dtype=torch.float64)
    labels = torch.tensor([label for _, label in pairs], dtype=torch.float64)
    probabilities = torch.sigmoid(float(model.admission_scale) * gaps + float(model.admission_bias))
    accepted = (gaps > 0) & (probabilities >= threshold)
    return {
        "pairs": len(pairs),
        "brier": float((probabilities - labels).square().mean()),
        "log_loss": float(F.binary_cross_entropy(probabilities, labels)),
        "admitted": int(accepted.sum()),
        "incorrect_admissions": int((accepted & (labels == 0)).sum()),
        "admission_precision": float(labels[accepted].mean()) if accepted.any() else None,
    }


def fit_admission(model, pairs):
    if len(pairs) < 4 or {label for _, label in pairs} != {0, 1}:
        raise ValueError("admission fitting needs at least four reviewed pairs and both outcomes")
    gaps = torch.tensor([gap for gap, _ in pairs], dtype=torch.float64)
    labels = torch.tensor([label for _, label in pairs], dtype=torch.float64)
    if not torch.isfinite(gaps).all():
        raise ValueError("non-finite admission utility gap")
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_scale, bias], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            log_scale.clamp(-4, 4).exp() * gaps + bias.clamp(-10, 10), labels
        )
        loss = loss + 0.001 * (log_scale.square() + bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    model.admission_scale.fill_(float(log_scale.detach().clamp(-4, 4).exp()))
    model.admission_bias.fill_(float(bias.detach().clamp(-10, 10)))
    model.admission_calibrated.fill_(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "fit"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--validation-reviews", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-generate", type=int, default=3)
    parser.add_argument("--proposal-branches", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite admission output")
    validate_suite(args.suite)
    scenarios = load_split(args.suite, "development")
    checkpoint_directory = resolve_checkpoint(args.checkpoint)
    engine = Engine.from_checkpoint(checkpoint_directory, args.device, max_new_tokens=args.max_new_tokens)
    source_fingerprint = fingerprint(checkpoint_directory)
    if args.command == "collect":
        options = GenerationOptions(
            n_generate=args.n_generate,
            proposal_branches=args.proposal_branches,
            temperature=args.temperature,
            seed=args.seed,
        ).model_dump()
        records, failures = collect_pairs(engine, scenarios, source_fingerprint, options)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(row.model_dump_json() + "\n" for row in records))
        print(
            json.dumps(
                {
                    "pairs": len(records),
                    "failures": failures,
                    "review": "Label whether the generated action is strictly better than the named incumbent under the scenario rubric. Equal or worse means false.",
                }
            )
        )
        return
    if args.reviews is None:
        parser.error("fit requires --reviews")
    pairs, groups, policy = reviewed_pairs(args.reviews, scenarios, engine, source_fingerprint)
    validation = None
    if args.validation_reviews:
        validation, validation_groups, validation_policy = reviewed_pairs(
            args.validation_reviews, scenarios, engine, source_fingerprint
        )
        if groups & validation_groups or policy != validation_policy:
            raise ValueError("validation must use disjoint scenario groups and the same generation policy")
    fit_admission(engine.model, pairs)
    engine.model.admission_policy = policy
    # Metadata only: no second model needs to be loaded.
    metadata = json.loads((checkpoint_directory / "system_one_plus_parallel.json").read_text())
    report = {
        "fitted": True,
        "policy": policy,
        "source_fingerprint": source_fingerprint,
        "reviews_sha256": digest(args.reviews),
        "fit_groups": sorted(groups),
        "fit_metrics": metrics(engine.model, pairs, engine.acceptance_threshold),
        "validation_metrics": metrics(engine.model, validation, engine.acceptance_threshold),
        "validation_reviews_sha256": digest(args.validation_reviews) if args.validation_reviews else None,
    }
    metadata["admission_calibration"] = report
    save_checkpoint(engine.model, engine.tokenizer, args.out, metadata)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
