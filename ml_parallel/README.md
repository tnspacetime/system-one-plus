# System One+ with parallel questions

`ml_parallel/` is a standalone alternative to `ml/`. One Qwen backbone and shared
LoRA adapter encode a state once, batch question continuations against that prefix,
then batch candidate continuations against their own state/question prefixes. Each question has independent candidate utilities, its own
set-attention omission prediction, and an optional proposal branch.

```text
state (computed once)
  ├── question 1 → isolated candidates → utilities + omission
  ├── question 2 → isolated candidates → utilities + omission
  └── question 3 → isolated candidates → utilities + omission
```

This package is self-contained, with its own Python namespace, commands,
configuration, lockfile, and checkpoint identity. Read [DESIGN.md](DESIGN.md) for
the attention and cache contracts.

## Setup and verification

```bash
cd ml_parallel
uv sync --locked --extra serve --extra cloud
uv run ruff check .
uv run ruff format --check .
uv run sop-parallel-data validate data/smoke-v1
uv run sop-parallel-train --config configs/integrated.toml --dry-run
```

These commands check formatting, the frozen smoke-data manifest, and configuration
loading. They do not establish model quality or architectural correctness.

CUDA uses **explicit FlashAttention 2.8.3 kernels** with BF16 by default, for both
training and inference. On Linux x86-64 with an NVIDIA Ampere-or-newer GPU and a
CUDA development toolkit compatible with PyTorch 2.8.0, install the GPU extra:

```bash
uv sync --locked --extra flash --extra serve
```

CPU/MPS use the FP32 reference implementation. `attention_mode="auto"` requires
FlashAttention on CUDA and does not silently fall back. For explicit CUDA reference
diagnostics, use `--attention-mode reference --compute-precision fp32` when serving.
Select the target device at model/checkpoint loading so the frozen base weights
use the correct precision. `/v1/models` and training reports expose the resolved
backend and compute precision. This release does not certify CUDA execution.

## Serving multiple questions

```bash
uv run sop-parallel-serve --checkpoint runs/parallel-smoke/checkpoint --port 8010
```

`POST /v1/systemone` accepts a shared state and a question-ID mapping:

```json
{
  "state": "The printer's paper tray is empty. A replacement cartridge is already in stock.",
  "questions": {
    "repair": {
      "mode": "challenge",
      "instructions": "What should happen next to restore printing?",
      "choices": {"a": "Restart the printer.", "b": "Replace the cartridge."},
      "n_generate": 3
    },
    "purchase": {
      "mode": "choose",
      "instructions": "Should another cartridge be ordered now?",
      "choices": {"yes": "Order another cartridge.", "no": "Use existing stock."}
    },
    "inspection": {
      "mode": "suggest",
      "instructions": "What useful checks should the technician perform?",
      "n_generate": 2
    }
  }
}
```

- `choose` ranks supplied actions and never generates.
- `challenge` uses that question's omission probability to decide whether to propose
  actions, independently scores new actions, and admits predicted improvements.
- `suggest` generates without a supplied menu, then scores the proposals.

`mode` defaults to `choose`. Question IDs route outputs and are not model tokens.
Answers are returned under the same IDs in `answers`, with the existing choice
provenance, utility, probabilities, expansion status, and proposal diagnostics.
`usage.prefill_tokens` counts tokens across the assessment stages, including the state only once.
Top-level timings cover shared assessment, batched generation, and total work;
per-question timing dictionaries are empty because execution is shared. Probabilities normalize within each question.

Assessment runs in state, question, and candidate stages, using length-bucketed
batches. It is no longer one giant masked forward pass. Triggered questions decode
in bounded batches using preallocated caches, with finished rows removed at batch
completion checks. Each
question keeps its own grammar and budget. A proposal failure is reported on its
question and does not discard other answers. An unsuccessful challenge retains its
supplied ranking; it does not automatically abstain from choosing a supplied action.

The flat single-question endpoints `/v1/systemone/choose`, `/suggest`, and
`/challenge` remain available, as does a flat choose request to `/v1/systemone`.
The multi-question API is intentionally scoped to action decisions.

Limits: at most 64 questions, 255 candidates per question including the requested
proposal budget, 1,024 logical positions per question branch by default, and 4,096 total
unique input tokens per request (the retained `max_packed_tokens` setting). Oversized
requests fail validation rather than truncating text. `branch_batch_size` bounds
each length bucket independently, with a default of 16 rows. `branch_bucket_width=0`
uses exact lengths; a positive value enables tested, padded buckets without changing
logical token positions. No performance benefit has been measured yet.
Proposal cache capacity is separately bounded by `max_cache_tokens` (4,096 per row),
and padded decoding allocations by `max_decode_batch_tokens` (16,384 total slots).
The proposal cache includes every supplied candidate, not just the longest branch.

For multiple sampled searches, add these fields to a `suggest` or `challenge` request:

```json
{"n_generate": 6, "proposal_branches": 3, "temperature": 0.8, "seed": 42}
```

This runs three independent searches with two actions of budget each. Their outputs
are deduplicated and independently scored. `search` reports completed branches and
errors. The defaults remain one greedy continuation; repeated greedy branches are
rejected because they would repeat the same search.

Seeded streams are derived from `(seed, stable question ID, proposal branch)`.
Reordering questions, changing which other questions trigger generation, or splitting
the decoding batch preserves each question's stream. Different question IDs separate
the streams even when they share the same user seed. Single-question methods use
the default ID `question`; the Python model's `propose()` also accepts `question_id`.
The lower-level `propose_many()` requires matching, unique `question_ids`.
Seeded outputs can differ from versions that did not include the question ID.

Grammar state, structural masks, validity checks, and greedy/sampled token selection
stay on the device. Seeded CUDA sampling never transfers vocabulary scores to the
CPU. The host checks one small batch-status vector every `decode_check_interval`
tokens (default 8), then transfers completed sequences for parsing and decoding.
Finished rows may perform up to seven extra dummy steps before removal; their
outputs and cache lengths are frozen. Set `--decode-check-interval 1` for immediate
completion checks at the cost of a host synchronization each step. This is a bounded
check interval, not a claim that generation has no CPU interaction.

## Training and comparison

```bash
uv run sop-parallel-train --config configs/integrated.toml --out runs/parallel-smoke
```

The objectives remain scoring, omission, and proposal generation. Omission uses
context-conditioned set attention. There is no dedicated paraphrase loss or mean/max
menu pooling. The pretrained backbone and vocabulary weights stay frozen; one LoRA
adapter and the utility/omission modules train jointly.

The trainer uses the same reviewed scenario format and frozen splits as `ml/`.
Each row describes one question. Rows sharing a state can contain different
questions; keep related rows in the same split/group. Derived menu variants can
also share a prefill because their branches cannot read each other.

Within each optimizer step, the trainer groups exact same-state examples, up to
`max_questions_per_pass` (default 8), bounded by `max_packed_tokens` (4,096). Each
group executes in stages; each stage buckets continuations by prefix and suffix
length and uses at most `branch_batch_size` rows.
It preserves each objective's normalization over the original accumulation batch.
`accumulation_steps` still counts question examples, not backbone calls. When an
optimizer batch has no repeated state, there is no state-sharing saving. Reports
include shared-state group counts and shared versus separate input-token counts.
Prefix K/V tensors remain attached to autograd. `gradient_checkpointing` checkpoints
entire continuation batches with explicit prefix tensor inputs and fresh per-call
cache collectors; the inference-only cache machinery is not used in training. Development
calibration and the existing quality evaluator use isolated questions.

`configs/integrated.toml` pins Qwen2.5-0.5B-Instruct;
`configs/capacity-1.5b.toml` pins Qwen2.5-1.5B-Instruct. The copied 8/3/3 smoke suite
is an unchanged software fixture, not evidence of decision quality. Replace it with
reviewed research data for an actual capability experiment.

```bash
uv run sop-parallel-evaluate --checkpoint runs/parallel-smoke/checkpoint \
  --suite data/smoke-v1 --split development --out runs/evaluation
uv run sop-parallel-harvest collect --checkpoint runs/parallel-smoke/checkpoint \
  --suite data/smoke-v1 --split train --out data/local/proposals.jsonl
uv run sop-parallel-harvest validate data/local/proposals.jsonl \
  --suite data/smoke-v1 --split train
```

Independently review collected candidates before continuation. Set `init_checkpoint`
and `reviewed_proposals` in a config to continue training. Checkpoints use
`parallel_utility_set_attention_v1`, checkpoint version 4, encoding version 2,
execution version `staged_flash_v3`. The registered backend name `sop_staged_sdpa`
is retained for compatibility; it dispatches explicitly to FlashAttention on CUDA.
The checkpoint records precision, attention mode, completion-check interval, batching,
and cache budgets. Existing version-1, version-2, and version-3 `ml_parallel` weights
load into the staged runtime without retraining or changing the original files;
returned metadata records their original execution version. Their admission
calibration is cleared because the proposal decoder changed. Checkpoints from
other architectures are incompatible. Matching base weights, seeds, and head
dimensions permits controlled fresh-initialization comparisons.

The Modal launcher uses a distinct app and run volume:

```bash
uv run modal run modal_app.py::train \
  --config configs/integrated.toml --name parallel-smoke --dry-run
```

Removing `--dry-run` submits GPU work. The launcher defaults to L4 and builds the
FlashAttention extra in a CUDA 12.8 development image. T4 is unsupported by this
FlashAttention configuration. No cloud job is needed for linting or the dry run.
CUDA assessment and training use unpadded variable-length FlashAttention with
lower-right causal alignment. Inference uses its static-cache kernel with per-row
cache lengths. Padded buckets do not introduce a CUDA attention mask.
There is no request-wide dense mask. Persistent prefix segments share storage,
but assembling a contiguous KV batch for a kernel still creates temporary tensors.
CPU correctness has been checked locally. Actual CUDA correctness and image builds
remain to be verified on suitable hardware; performance benchmarking is deferred.

## Admission probabilities from reviewed generated actions

Ranking temperature no longer supplies admission probabilities. Until a separate
fit exists for the current generation policy, admission uses `admission_margin`
(default 1.3863 in raw utility units). Responses explicitly report `utility_margin`
and a null `admission_probability`; that margin is configurable, not a probability.

Collect actual, gate-triggered generated-vs-incumbent pairs from **development**:

```bash
uv run sop-parallel-calibrate-admission collect \
  --checkpoint runs/parallel-smoke/checkpoint --suite data/smoke-v1 \
  --out data/local/admission-review.jsonl
```

Review `better_than_incumbent` for every pair using its frozen scenario rubric.
True means strictly better than the named incumbent; equal or worse means false.
Use the same generation options as deployment when collecting. The collector also
reports decoding failures. Keep distinct scenario groups for fitting and validation.

```bash
uv run sop-parallel-calibrate-admission fit \
  --checkpoint runs/parallel-smoke/checkpoint --suite data/smoke-v1 \
  --reviews data/local/admission-fit.jsonl \
  --validation-reviews data/local/admission-validation.jsonl \
  --out runs/parallel-admission-calibrated
```

Validation is optional but its metrics stay null when omitted. A fit requires both
outcomes and at least four reviewed pairs; that minimum is a software guard, not a
claim of sufficient statistical evidence. Checkpoint, policy, split, and group
checks prevent accidental reuse of incompatible reviews. A changed generation
policy, attention backend, or compute precision falls back to the explicit utility
margin. Further training clears the fit.
The smoke suite does not establish real-world calibration.
