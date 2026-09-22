"""Explicit, single-job Modal launcher for System One Plus training. Importing never launches a job."""

import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REMOTE = Path("/workspace/ml_parallel")
app = modal.App("system-one-plus-parallel-training")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"MAX_JOBS": "2"})
    .uv_sync(uv_project_dir=str(ROOT), groups=[], extras=["flash"], uv_version="0.9.13")
    .env({"HF_HOME": "/hf", "PYTHONPATH": str(REMOTE), "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "system_one_plus_parallel", str(REMOTE / "system_one_plus_parallel"))
    .add_local_dir(ROOT / "configs", str(REMOTE / "configs"))
    .add_local_dir(ROOT / "data", str(REMOTE / "data"))
)
cache = modal.Volume.from_name("system-one-plus-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("system-one-plus-parallel-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=16384,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/hf": cache, "/runs": runs},
)
def train_remote(
    config_name: str, suite_relative: str, name: str, max_steps: int | None, expected_hashes: dict
):
    from system_one_plus_parallel.config import TrainingConfig, load_config
    from system_one_plus_parallel.data import digest
    from system_one_plus_parallel.train import source_hashes, train

    config, _ = load_config(REMOTE / "configs" / config_name)
    actual = source_hashes(REMOTE)
    actual["config"] = digest(REMOTE / "configs" / config_name)
    actual["suite"] = digest(REMOTE / suite_relative / "manifest.json")
    if config.reviewed_proposals:
        actual["reviewed_proposals"] = digest(Path(config.reviewed_proposals))
    if actual != expected_hashes:
        raise ValueError("local and remote source/config/suite hashes differ")
    changes = {"device": "cuda", **({"max_steps": max_steps} if max_steps is not None else {})}
    config = TrainingConfig.model_validate({**config.model_dump(), **changes})
    try:
        return train(config, REMOTE / suite_relative, Path("/runs") / name)
    finally:
        cache.commit()
        runs.commit()


@app.local_entrypoint()
def train(
    config: str,
    name: str,
    suite: str = "",
    gpu: str = "L4",
    max_steps: int = 0,
    timeout: int = 1800,
    dry_run: bool = False,
):
    """Launch one bounded job; --dry-run validates and prints the plan without submitting GPU work."""
    from system_one_plus_parallel.config import load_config
    from system_one_plus_parallel.data import digest, validate_suite
    from system_one_plus_parallel.model import ARCHITECTURE
    from system_one_plus_parallel.train import source_hashes

    path = (ROOT / config).resolve()
    if path.parent != ROOT / "configs" or path.suffix != ".toml":
        raise ValueError("config must be a TOML file immediately under ml_parallel/configs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("name must be a simple run identifier")
    if gpu not in {"L4", "A10G", "A100", "H100"} or not 60 <= timeout <= 7200 or max_steps < 0:
        raise ValueError("unsupported GPU, timeout outside 60..7200 seconds, or negative max_steps")
    cfg, suite_path = load_config(path)
    suite_path = (ROOT / suite).resolve() if suite else suite_path
    relative = suite_path.relative_to(ROOT / "data")
    suite_relative = str(Path("data") / relative)
    manifest = validate_suite(suite_path)
    hashes = source_hashes(ROOT)
    hashes.update(config=digest(path), suite=digest(suite_path / "manifest.json"))
    if cfg.reviewed_proposals:
        Path(cfg.reviewed_proposals).relative_to(ROOT / "data")
        hashes["reviewed_proposals"] = digest(Path(cfg.reviewed_proposals))
    plan = {
        "architecture": ARCHITECTURE,
        "gpu": gpu,
        "timeout_seconds": timeout,
        "max_steps": max_steps or cfg.max_steps,
        "suite": manifest["name"],
        "purpose": manifest["purpose"],
        "remote_output": f"/runs/{name}",
    }
    print(json.dumps(plan, indent=2))
    if dry_run:
        return
    report = train_remote.with_options(gpu=gpu, timeout=timeout).remote(
        path.name, suite_relative, name, max_steps or None, hashes
    )
    print(json.dumps(report, indent=2))
    print(f"Download: uv run modal volume get system-one-plus-parallel-runs /{name} runs/")
