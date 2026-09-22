"""Local HTTP interface to one loaded independent-utility model."""

import argparse
import logging
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import device_for
from .engine import Engine
from .errors import InvalidRequestError, ProposalOutputError
from .model import ARCHITECTURE
from .schema import GenerationOptions, ParallelRequest

app = FastAPI(title="System One+ parallel questions")
ENGINE = None
LOCK = threading.RLock()
logger = logging.getLogger(__name__)


class SuggestRequest(GenerationOptions):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    state: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    n_generate: int = Field(default=3, ge=1, le=32)
    top_k: int | None = Field(default=None, ge=1, le=255)


class ChooseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    state: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    choices: dict[str, str] = Field(min_length=1, max_length=255)
    top_k: int | None = Field(default=None, ge=1, le=255)


class ChallengeRequest(ChooseRequest, GenerationOptions):
    n_generate: int = Field(default=3, ge=1, le=32)


def invoke(request, method):
    if ENGINE is None:
        raise HTTPException(503, "model is not loaded")
    kwargs = request.model_dump()
    if "instructions" in kwargs:
        kwargs["question"] = kwargs.pop("instructions")
    try:
        with LOCK:
            return getattr(ENGINE, method)(**kwargs)
    except InvalidRequestError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ProposalOutputError as exc:
        raise HTTPException(502, {"status": exc.status, "message": str(exc)}) from exc
    except Exception as exc:
        logger.exception("model inference failed")
        raise HTTPException(500, "model inference failed") from exc


@app.post("/v1/systemone")
def systemone(request: ParallelRequest | ChooseRequest):
    return invoke(request, "answer" if isinstance(request, ParallelRequest) else "choose")


@app.post("/v1/systemone/choose")
def choose(request: ChooseRequest):
    return invoke(request, "choose")


@app.post("/v1/systemone/suggest")
def suggest(request: SuggestRequest):
    return invoke(request, "suggest")


@app.post("/v1/systemone/challenge")
def challenge(request: ChallengeRequest):
    return invoke(request, "challenge")


@app.get("/v1/models")
def models():
    return {
        "models": [
            {
                "id": "system-one-plus-parallel",
                "architecture": ARCHITECTURE,
                "loaded": ENGINE is not None,
                **(ENGINE.model.runtime if ENGINE else {"attention": None, "compute_precision": None}),
            }
        ]
    }


def main():
    global ENGINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attention-mode", choices=["auto", "flash", "reference"])
    parser.add_argument("--compute-precision", choices=["auto", "fp32", "bf16", "fp16"])
    parser.add_argument("--decode-check-interval", type=int)
    parser.add_argument("--branch-bucket-width", type=int)
    parser.add_argument("--max-cache-tokens", type=int)
    parser.add_argument("--max-decode-batch-tokens", type=int)
    parser.add_argument("--admission-margin", type=float)
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()
    overrides = {"max_new_tokens": args.max_new_tokens}
    for name in (
        "branch_bucket_width",
        "max_cache_tokens",
        "max_decode_batch_tokens",
        "admission_margin",
        "attention_mode",
        "compute_precision",
        "decode_check_interval",
    ):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    if args.threshold is not None:
        overrides["expansion_threshold"] = args.threshold
    ENGINE = Engine.from_checkpoint(
        args.checkpoint, device_for(args.device), revision=args.checkpoint_revision, **overrides
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
