"""Validate frozen suites and derive joint System One+ supervision."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .schema import Scenario, normalized

SPLITS = ("train", "development", "test")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_scenarios(path: Path) -> list[Scenario]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(Scenario.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty partition: {path}")
    return rows


def validate_partitions(directory: Path) -> dict:
    seen_ids, seen_groups, seen_states = set(), {}, {}
    files = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        rows = read_scenarios(path)
        for row in rows:
            if row.id in seen_ids:
                raise ValueError(f"duplicate scenario id: {row.id}")
            seen_ids.add(row.id)
            for key, seen, kind in (
                (row.group_id, seen_groups, "group"),
                (normalized(row.state), seen_states, "state"),
            ):
                if key in seen and seen[key] != split:
                    raise ValueError(f"{kind} leakage between {seen[key]} and {split}: {row.id}")
                seen[key] = split
        files[path.name] = {"sha256": digest(path), "records": len(rows)}
    return files


def validate_suite(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported suite schema_version")
    if validate_partitions(directory) != manifest["files"]:
        raise ValueError("frozen suite checksum/count mismatch; create a new suite version")
    return manifest


def load_split(directory: Path, split: str, *, allow_test: bool = False) -> list[Scenario]:
    if split not in SPLITS or (split == "test" and not allow_test):
        raise ValueError("test evaluation requires --allow-test; training uses train only")
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}.jsonl"
    rows = read_scenarios(path)
    if manifest["files"][path.name] != {"sha256": digest(path), "records": len(rows)}:
        raise ValueError(f"frozen partition mismatch: {path}")
    return rows


def unique_actions(values):
    seen, result = set(), []
    for value in values:
        key = normalized(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def joint_examples(rows, seed=0, reviewed=None):
    """Expand reviewed single-best scenario families without split leakage.

    Accepted reviews mean best-quality repairs under the scenario rubric. They
    are ranking positives, not automatically paraphrases of the canonical answer.
    """
    randomizer = random.Random(seed)
    reviewed = reviewed or {}
    examples = []
    for row in rows:
        policy = reviewed.get(row.id, {"accepted": [], "rejected": []})
        canonical = unique_actions([row.best_choice, *row.acceptable_equivalents])
        good = unique_actions([*canonical, *policy["accepted"]])
        bad = unique_actions(
            [c for c in row.complete_choices if c != row.best_choice]
            + row.rejected_proposals
            + policy["rejected"]
        )
        if {normalized(a) for a in good} & {normalized(a) for a in bad}:
            raise ValueError(f"conflicting action quality labels for {row.id}")
        targets = unique_actions([row.best_choice, *policy["accepted"]])[:3]
        irrelevant_removed = [row.best_choice] + [c for c in row.complete_choices if c != row.best_choice][1:]
        variants = [
            ("complete", row.complete_choices, 0.0, None),
            ("incomplete", row.incomplete_choices, 1.0, targets),
            ("irrelevant_removed", irrelevant_removed, 0.0, None),
            ("expanded", unique_actions([*row.complete_choices, *bad, *good]), 0.0, None),
            ("suggest", [], None, targets),
        ]
        if row.acceptable_equivalents:
            variants.append(
                ("paraphrase_retained", [row.acceptable_equivalents[0], *row.incomplete_choices], 0.0, None)
            )
        for variant, options, omission, proposals in variants:
            choices = list(options)
            randomizer.shuffle(choices)
            keys = [normalized(c) for c in choices]
            positives = [i for i, c in enumerate(keys) if c in {normalized(a) for a in good}]
            negatives = [i for i, c in enumerate(keys) if c in {normalized(a) for a in bad}]
            examples.append(
                {
                    "id": f"{row.id}/{variant}",
                    "scenario_id": row.id,
                    "variant": variant,
                    "state": row.state,
                    "question": row.question,
                    "choices": choices,
                    "preferences": [(a, b) for a in positives for b in negatives],
                    "omission_target": omission,
                    "proposal_target": proposals,
                }
            )
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument("suite", type=Path)
    freeze = commands.add_parser("freeze", help="validate and checksum a new suite version")
    freeze.add_argument("suite", type=Path)
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--purpose", choices=["smoke", "research"], required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        manifest = args.suite / "manifest.json"
        if manifest.exists():
            raise FileExistsError("refusing to replace a frozen manifest; use a new version")
        write_json(
            manifest,
            {
                "schema_version": 1,
                "name": args.name,
                "purpose": args.purpose,
                "files": validate_partitions(args.suite),
            },
        )
    print(json.dumps(validate_suite(args.suite), indent=2))


if __name__ == "__main__":
    main()
