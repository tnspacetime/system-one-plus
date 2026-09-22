"""Evaluate capability, restraint, order sensitivity, failures, and latency."""

import argparse
import json
import random
from pathlib import Path

from .config import device_for
from .data import load_split, validate_suite, write_json
from .engine import Engine
from .errors import ProposalOutputError
from .model import load_checkpoint
from .schema import normalized


def accepted(row, text: str) -> bool:
    return normalized(text) in {
        normalized(row.best_choice),
        *(normalized(value) for value in row.acceptable_equivalents),
    }


def menu(values: list[str]) -> dict[str, str]:
    return {f"c{index}": value for index, value in enumerate(values)}


def selected(response: dict | None):
    if response is None:
        return None
    identifier = response["selected_choice_id"]
    return next((choice for choice in response["choices"] if choice["id"] == identifier), None)


def selected_is_accepted(row, response: dict | None) -> bool:
    choice = selected(response)
    return choice is not None and accepted(row, choice["text"])


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else None


def permutation_orders(identifiers: list[str], count: int, seed: int) -> list[list[str]]:
    if not 1 <= count <= 32:
        raise ValueError("permutations must be between 1 and 32")
    if len(identifiers) < 2:
        return [list(identifiers)]
    orders, seen = [], set()

    def add(order):
        key = tuple(order)
        if key not in seen:
            orders.append(list(order))
            seen.add(key)

    for shift in range(min(len(identifiers), count)):
        add(identifiers[shift:] + identifiers[:shift])
    randomizer = random.Random(seed)
    attempts = 0
    while len(orders) < count and attempts < count * 20:
        order = list(identifiers)
        randomizer.shuffle(order)
        add(order)
        attempts += 1
    return orders


def winner_ids(probabilities: dict[str, float]) -> list[str]:
    maximum = max(probabilities.values())
    return sorted(
        identifier for identifier, probability in probabilities.items() if abs(probability - maximum) <= 1e-12
    )


def order_diagnostics(
    engine: Engine,
    state: str,
    question: str,
    choices: dict[str, str],
    *,
    permutations: int,
    seed: int,
):
    runs = []
    for order in permutation_orders(list(choices), permutations, seed):
        ordered = {identifier: choices[identifier] for identifier in order}
        result = engine.diagnose_ordered_menu(state, question, ordered)
        runs.append(
            {
                "order": order,
                "winner_ids": winner_ids(result["probabilities"]),
                "probabilities": result["probabilities"],
                "expansion_probability": result["expansion_probability"],
                "utilities": result["utilities"],
                "expansion_selected": result["expansion_probability"] >= engine.expansion_threshold,
            }
        )
    first_winners = runs[0]["winner_ids"]
    expansion_probabilities = [run["expansion_probability"] for run in runs]
    return {
        "permutations": len(runs),
        "choice_winner_stable": len(first_winners) == 1
        and all(run["winner_ids"] == first_winners for run in runs),
        "gate_decision_stable": len({run["expansion_selected"] for run in runs}) == 1,
        "gate_probability_spread": max(expansion_probabilities) - min(expansion_probabilities),
        "max_utility_spread": max(
            max(run["utilities"][k] for run in runs) - min(run["utilities"][k] for run in runs)
            for k in choices
        ),
        "runs": runs,
    }


def safe_challenge(engine: Engine, state: str, question: str, choices: dict[str, str]):
    try:
        response = engine.challenge(state, question, choices)
        error = response["expansion"].get("error")
        return response, ({"message": error} if error else None)
    except ProposalOutputError as exc:
        return None, {"type": type(exc).__name__, "message": str(exc)}


def response_menu(response: dict | None) -> dict[str, str] | None:
    if response is None:
        return None
    return {choice["id"]: choice["text"] for choice in response["choices"]}


def aggregate_order(predictions: list[dict], condition: str):
    diagnostics = [row[condition] for row in predictions if row.get(condition) is not None]
    if not diagnostics:
        return None
    return {
        "records": len(diagnostics),
        "choice_winner_stable_rate": sum(item["choice_winner_stable"] for item in diagnostics)
        / len(diagnostics),
        "gate_decision_stable_rate": sum(item["gate_decision_stable"] for item in diagnostics)
        / len(diagnostics),
        "max_utility_spread": max(item["max_utility_spread"] for item in diagnostics),
        "mean_gate_probability_spread": sum(item["gate_probability_spread"] for item in diagnostics)
        / len(diagnostics),
    }


def mean_latency(predictions: list[dict], response_key: str):
    values = [
        row[response_key]["latency_ms"]["total"] for row in predictions if row[response_key] is not None
    ]
    return sum(values) / len(values) if values else None


def evaluate(engine: Engine, rows, *, permutations: int = 8, permutation_seed: int = 42):
    predictions = []
    for index, row in enumerate(rows):
        complete = menu(row.complete_choices)
        incomplete = menu(row.incomplete_choices)
        original = engine.choose(row.state, row.question, complete)
        repaired, incomplete_error = safe_challenge(engine, row.state, row.question, incomplete)
        retained, complete_error = safe_challenge(engine, row.state, row.question, complete)
        generated = repaired.get("proposals", []) if repaired else []
        repaired_selection = selected(repaired)
        row_seed = permutation_seed + index * 1009
        repaired_choices = response_menu(repaired)
        retained_choices = response_menu(retained)
        predictions.append(
            {
                "id": row.id,
                "rubric": row.rubric,
                "best_choice": row.best_choice,
                "complete_correct": selected_is_accepted(row, original),
                "incomplete_detected": (
                    repaired["expansion"]["attempted"] if repaired else incomplete_error is not None
                ),
                "complete_expanded": (
                    retained["expansion"]["attempted"] if retained else complete_error is not None
                ),
                "proposal_failed": incomplete_error is not None,
                "complete_proposal_failed": complete_error is not None,
                "exact_reference_recovered": any(accepted(row, choice["text"]) for choice in generated),
                "end_to_end_repair": selected_is_accepted(row, repaired)
                and bool(repaired_selection and repaired_selection["source"] == "generated"),
                "generated_action_won": bool(
                    repaired_selection and repaired_selection["source"] == "generated"
                ),
                "original_good_displaced": retained is not None
                and selected_is_accepted(row, original)
                and not selected_is_accepted(row, retained),
                "choose_response": original,
                "incomplete_response": repaired,
                "complete_response": retained,
                "incomplete_proposal_error": incomplete_error,
                "complete_proposal_error": complete_error,
                "complete_order": order_diagnostics(
                    engine,
                    row.state,
                    row.question,
                    complete,
                    permutations=permutations,
                    seed=row_seed,
                ),
                "incomplete_order": order_diagnostics(
                    engine,
                    row.state,
                    row.question,
                    incomplete,
                    permutations=permutations,
                    seed=row_seed + 1,
                ),
                "repaired_order": order_diagnostics(
                    engine,
                    row.state,
                    row.question,
                    repaired_choices,
                    permutations=permutations,
                    seed=row_seed + 2,
                )
                if repaired_choices
                else None,
                "retained_order": order_diagnostics(
                    engine,
                    row.state,
                    row.question,
                    retained_choices,
                    permutations=permutations,
                    seed=row_seed + 3,
                )
                if retained_choices
                else None,
                "forced_proposal": engine.propose_for_review(row.state, row.question, incomplete),
                "review_required": True,
            }
        )
    complete_correct = [row for row in predictions if row["complete_correct"]]
    recovered = [row for row in predictions if row["exact_reference_recovered"]]
    gate_rows = [
        (response["expansion"]["probability"], label)
        for prediction in predictions
        for key, label in (("incomplete_response", 1), ("complete_response", 0))
        if (response := prediction[key]) is not None
    ]
    true_positive = sum(p >= engine.expansion_threshold and label == 1 for p, label in gate_rows)
    triggered = sum(p >= engine.expansion_threshold for p, _ in gate_rows)
    report = {
        "records": len(predictions),
        "gate_brier": sum((p - label) ** 2 for p, label in gate_rows) / len(gate_rows) if gate_rows else None,
        "omission_precision": true_positive / triggered if triggered else None,
        "complete_menu_accuracy": mean(predictions, "complete_correct"),
        "omission_recall": mean(predictions, "incomplete_detected"),
        "unnecessary_expansion_rate": mean(predictions, "complete_expanded"),
        "proposal_failure_rate": mean(predictions, "proposal_failed"),
        "complete_proposal_failure_rate": mean(predictions, "complete_proposal_failed"),
        "exact_reference_recovery_rate": mean(predictions, "exact_reference_recovered"),
        "end_to_end_repair_rate": mean(predictions, "end_to_end_repair"),
        "selection_given_recovery": mean(recovered, "end_to_end_repair"),
        "reference_displacement_rate": mean(complete_correct, "original_good_displaced"),
        "order_sensitivity": {
            "permutations_requested": permutations,
            "seed": permutation_seed,
            "complete": aggregate_order(predictions, "complete_order"),
            "incomplete": aggregate_order(predictions, "incomplete_order"),
            "repaired": aggregate_order(predictions, "repaired_order"),
            "retained": aggregate_order(predictions, "retained_order"),
        },
        "mean_latency_ms": {
            "choose": mean_latency(predictions, "choose_response"),
            "incomplete_challenge": mean_latency(predictions, "incomplete_response"),
            "complete_challenge": mean_latency(predictions, "complete_response"),
        },
        "forced_reference_recovery_rate": sum(
            any(accepted(source, action) for action in prediction["forced_proposal"]["candidates"])
            for source, prediction in zip(rows, predictions, strict=True)
        )
        / len(rows),
        "matching": "reference proxies only: normalized reviewed aliases; independently review novel generated actions",
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=8, choices=range(1, 33))
    parser.add_argument("--permutation-seed", type=int, default=42)
    args = parser.parse_args()
    manifest = validate_suite(args.suite)
    rows = load_split(args.suite, args.split, allow_test=args.allow_test)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite evaluation results")
    tokenizer, model, metadata = load_checkpoint(
        args.checkpoint,
        device_for(args.device),
        revision=args.checkpoint_revision,
    )
    threshold = metadata["expansion_threshold"] if args.threshold is None else args.threshold
    report, predictions = evaluate(
        Engine(
            tokenizer,
            model,
            expansion_threshold=threshold,
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
        permutations=args.permutations,
        permutation_seed=args.permutation_seed,
    )
    args.out.mkdir(parents=True)
    write_json(
        args.out / "report.json",
        {
            **report,
            "suite": manifest["name"],
            "purpose": manifest["purpose"],
            "split": args.split,
            "expansion_threshold": threshold,
        },
    )
    (args.out / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
