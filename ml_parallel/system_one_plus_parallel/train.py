"""Joint utility, omission, and proposal training on one LoRA adapter."""

import argparse
import contextlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import torch

from .calibration import fit_calibration
from .config import TrainingConfig, device_for, load_config
from .data import digest, joint_examples, load_split, validate_suite, write_json
from .encoding import Codec, ParallelEncoding
from .harvest import load_reviewed_proposals
from .model import ARCHITECTURE, SystemOnePlusModel, load_checkpoint, load_tokenizer, save_checkpoint

COMPONENTS = ("score", "omission", "generation")


def source_hashes(root):
    return {
        str(p.relative_to(root)): digest(p) for p in sorted((root / "system_one_plus_parallel").glob("*.py"))
    }


def prepare(config, rows, codec, reviewed=None):
    return [
        (
            example,
            codec.encode(
                example["state"],
                example["question"],
                example["choices"],
                targets=example["proposal_target"],
                max_state=config.max_state_tokens,
                max_length=config.max_length,
                max_work_tokens=config.max_packed_tokens,
                max_cache_tokens=config.max_cache_tokens,
            ),
        )
        for example in joint_examples(rows, config.seed, reviewed)
    ]


def losses_for(model, example, encoding):
    return model.losses(
        encoding,
        preferences=example["preferences"],
        omission_target=example["omission_target"],
    )


def pack_batch(batch, codec, config):
    """Group exact shared states within an optimizer batch; never change loss weighting."""
    groups = {}
    for index, (example, encoding) in enumerate(batch):
        state_end = encoding.prefix.index(codec.ids["question"])
        state = encoding.prefix[:state_end]
        groups.setdefault((example["state"], state), []).append((str(index), example, encoding))
    packs = []
    for (_, state), rows in groups.items():
        encodings, examples, size = {}, {}, len(state)
        for key, example, encoding in rows:
            extra = encoding.size - len(state)
            if encodings and (
                size + extra > config.max_packed_tokens or len(encodings) >= config.max_questions_per_pass
            ):
                packs.append((ParallelEncoding(state, encodings), examples))
                encodings, examples, size = {}, {}, len(state)
            if size + extra > config.max_packed_tokens:
                raise ValueError("training example exceeds packed token budget")
            encodings[key], examples[key] = encoding, example
            size += extra
        packs.append((ParallelEncoding(state, encodings), examples))
    return packs


@torch.no_grad()
def development_metrics(model, examples):
    model.eval()
    components = {name: [] for name in COMPONENTS}
    gates, gaps = [], []
    for example, encoding in examples:
        losses = losses_for(model, example, encoding)
        if any(not torch.isfinite(loss) for loss in losses.values()):
            raise RuntimeError("non-finite development loss")
        for name, value in losses.items():
            components[name].append(float(value))
        result = model(encoding)
        if example["omission_target"] is not None:
            gates.append((float(result["omission_logit"]), int(example["omission_target"])))
        # Use the complete condition once per scenario for calibration.
        if example["variant"] == "complete":
            gaps.extend(
                float(result["utilities"][a] - result["utilities"][b]) for a, b in example["preferences"]
            )
    return {
        "loss": {k: sum(v) / len(v) if v else None for k, v in components.items()},
        "gate_rows": gates,
        "preference_differences": gaps,
    }


def train(config, suite, out):
    manifest = validate_suite(suite)
    training_rows, development_rows = load_split(suite, "train"), load_split(suite, "development")
    reviewed, review_stats = (
        load_reviewed_proposals(Path(config.reviewed_proposals), training_rows)
        if config.reviewed_proposals
        else ({}, None)
    )
    if out.exists():
        raise FileExistsError(f"refusing to overwrite run: {out}")
    device = device_for(config.device)
    if config.dtype == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires CUDA with BF16 support")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    run = {
        "architecture": ARCHITECTURE,
        "status": "running",
        "device": device,
        "config": config.model_dump(),
        "suite_manifest_sha256": digest(suite / "manifest.json"),
        "suite_purpose": manifest["purpose"],
        "source_sha256": source_hashes(root),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        "review_sha256": digest(Path(config.reviewed_proposals)) if config.reviewed_proposals else None,
    }
    write_json(out / "run.json", run)
    started = time.perf_counter()
    try:
        if config.init_checkpoint:
            tokenizer, model, parent = load_checkpoint(
                config.init_checkpoint,
                device,
                revision=config.init_checkpoint_revision,
                trainable=True,
                attention_mode=config.attention_mode,
                compute_precision=config.dtype,
            )
            if (parent["base_model"], parent["base_revision"]) != (config.base_model, config.base_revision):
                raise ValueError("parent model/revision differs from training config")
            dimensions = ("lora_rank", "head_dim", "omission_heads", "omission_layers")
            if any(getattr(model, name) != getattr(config, name) for name in dimensions):
                raise ValueError("parent head/LoRA dimensions differ from training config")
            initialization = {"kind": "utility_checkpoint", "parent": config.init_checkpoint}
        else:
            tokenizer = load_tokenizer(config.base_model, config.base_revision)
            model = SystemOnePlusModel(
                config.base_model,
                revision=config.base_revision,
                lora_rank=config.lora_rank,
                head_dim=config.head_dim,
                omission_heads=config.omission_heads,
                omission_layers=config.omission_layers,
                branch_batch_size=config.branch_batch_size,
                device=device,
                attention_mode=config.attention_mode,
                compute_precision=config.dtype,
                decode_check_interval=config.decode_check_interval,
            )
            initialization = {"kind": "qwen", "parent": None}
        codec = Codec(tokenizer)
        model.branch_batch_size = config.branch_batch_size
        model.decode_check_interval = config.decode_check_interval
        model.branch_bucket_width = config.branch_bucket_width
        run["runtime"] = model.runtime
        if model.runtime["attention"] == "flash_attention_2":
            run["versions"]["flash-attn"] = importlib.metadata.version("flash-attn")
        write_json(out / "run.json", run)
        # Updating weights invalidates calibration fitted for the parent checkpoint.
        model.admission_calibrated.fill_(False)
        model.admission_scale.fill_(1.0)
        model.admission_bias.zero_()
        model.gradient_checkpointing = config.gradient_checkpointing
        parameters = model.trainable_parameters()
        development = prepare(config, development_rows, codec)
        examples = prepare(config, training_rows, codec, reviewed)
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        # The model controls backbone precision in training and evaluation alike.
        autocast = contextlib.nullcontext
        initial = development_metrics(model, development)
        totals, counts = dict.fromkeys(COMPONENTS, 0.0), dict.fromkeys(COMPONENTS, 0)
        weights = {name: getattr(config, f"{name}_weight") for name in COMPONENTS}
        steps = seen = state_groups = prefill_tokens = separate_prefill_tokens = 0
        with (out / "metrics.jsonl").open("w") as log:
            for epoch in range(config.epochs):
                random.Random(config.seed + epoch).shuffle(examples)
                model.train()
                for start in range(0, len(examples), config.accumulation_steps):
                    batch = examples[start : start + config.accumulation_steps]
                    # Normalize each objective over its active examples in this optimizer step.
                    active = {
                        name: sum(
                            bool(
                                e["preferences"]
                                if name == "score"
                                else e["omission_target"] is not None
                                if name == "omission"
                                else e["proposal_target"] is not None
                            )
                            for e, _ in batch
                        )
                        for name in COMPONENTS
                    }
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    packs = pack_batch(batch, codec, config)
                    for encoding, grouped in packs:
                        with autocast():
                            question_losses = model.losses_questions(
                                encoding,
                                {
                                    key: {
                                        "preferences": e["preferences"],
                                        "omission_target": e["omission_target"],
                                    }
                                    for key, e in grouped.items()
                                },
                            )
                            loss = sum(
                                weights[k] * value / active[k]
                                for losses in question_losses.values()
                                for k, value in losses.items()
                            )
                        if not torch.isfinite(loss):
                            raise RuntimeError("non-finite training loss")
                        loss.backward()
                        total += float(loss.detach())
                        for losses in question_losses.values():
                            for key, value in losses.items():
                                totals[key] += float(value.detach())
                                counts[key] += 1
                        state_groups += 1
                        prefill_tokens += encoding.size
                        separate_prefill_tokens += sum(q.size for q in encoding.questions.values())
                    adapter_norm = math.sqrt(
                        sum(
                            float(p.grad.detach().float().square().sum())
                            for name, p in model.named_parameters()
                            if "lora_" in name and p.grad is not None
                        )
                    )
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    seen += len(batch)
                    entry = {
                        "step": steps,
                        "loss": total,
                        "gradient_norm": float(norm),
                        "adapter_gradient_norm": adapter_norm,
                        "shared_state_groups": len(packs),
                        "questions": len(batch),
                    }
                    log.write(json.dumps(entry, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(entry), flush=True)
                    if config.max_steps and steps >= config.max_steps:
                        break
                if config.max_steps and steps >= config.max_steps:
                    break
        final = development_metrics(model, development)
        threshold, calibration = fit_calibration(
            model, final.pop("gate_rows"), final.pop("preference_differences")
        )
        initial.pop("gate_rows")
        initial.pop("preference_differences")
        metadata = {
            "expansion_threshold": threshold,
            "acceptance_threshold": config.acceptance_threshold,
            "max_state_tokens": config.max_state_tokens,
            "max_length": config.max_length,
            "max_packed_tokens": config.max_packed_tokens,
            "max_cache_tokens": config.max_cache_tokens,
            "max_decode_batch_tokens": config.max_decode_batch_tokens,
            "admission_margin": config.admission_margin,
            "admission_calibration": {"fitted": False},
            "calibration": calibration,
            "provenance": run,
            "initialization": initialization,
            "acceptance_threshold_source": "configured; validate on independently reviewed proposals",
        }
        save_checkpoint(model, tokenizer, out / "checkpoint", metadata)
        report = {
            "steps": steps,
            "examples_seen": seen,
            "training_shared_state_groups": state_groups,
            "training_prefill_tokens": prefill_tokens,
            "separate_prefill_tokens": separate_prefill_tokens,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "component_train_loss": {k: totals[k] / counts[k] if counts[k] else None for k in COMPONENTS},
            "development_before": initial,
            "development_after": final,
            "expansion_threshold": threshold,
            "calibration": calibration,
            "initialization": initialization,
            "reviewed_proposals": review_stats,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(out / "report.json", report)
        write_json(out / "run.json", {**run, "status": "complete", "report": report})
        return report
    except Exception as exc:
        write_json(out / "run.json", {**run, "status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, suite = load_config(args.config)
    config = TrainingConfig.model_validate(
        {
            **config.model_dump(),
            **{
                k: v for k, v in {"device": args.device, "max_steps": args.max_steps}.items() if v is not None
            },
        }
    )
    suite = args.suite.resolve() if args.suite else suite
    manifest = validate_suite(suite)
    if args.dry_run:
        reviewed = (
            load_reviewed_proposals(Path(config.reviewed_proposals), load_split(suite, "train"))[0]
            if config.reviewed_proposals
            else {}
        )
        count = len(joint_examples(load_split(suite, "train"), reviewed=reviewed))
        steps = config.epochs * math.ceil(count / config.accumulation_steps)
        print(
            json.dumps(
                {
                    "architecture": ARCHITECTURE,
                    "suite_purpose": manifest["purpose"],
                    "training_examples": count,
                    "optimizer_steps": min(steps, config.max_steps or steps),
                    "config": config.model_dump(),
                },
                indent=2,
            )
        )
        return
    if args.out is None:
        parser.error("--out is required unless --dry-run is used")
    train(config, suite, args.out)


if __name__ == "__main__":
    main()
