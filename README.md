# System One+

> The menu is not the world.

[Jev](https://typesafe.ai/) makes the case for System One models: fast, typed
decisions with probabilities instead of autoregressive text. System One+ starts
from that achievement and asks the next question: if software acts on Jev's
scores, what happens when the supplied action menu is incomplete?

System One+ is an open research proposal for decision systems that do more than
select the best option they were given. It asks a harder question:

**Can we build an efficient architecture that detects when an action menu may be
incomplete and, when needed, surfaces what should be added?**

This repository contains an executable research prototype of the architecture.

**[Read the proposal and specification →](https://system-one-plus.tnspacetime.com)**

## Launch film

https://github.com/user-attachments/assets/30ec01cd-ac6c-4394-a639-d58e928e09b9

The **2:53 launch film** presents the argument in motion.

## The case

A decision score is not produced for fun. Software acts on it: it invokes a tool,
routes a payment, changes access, restarts a service, or escalates a person. The
score becomes an action, and the action has real consequences.

You do not rely on a decision model because you already know which action is best.
You rely on it because you trust its learned judgment enough to act on the result.
The objective is therefore not merely to select the highest-scoring item in a menu.
It is to take the best action available.

Closed-set decision systems assume that the useful action is already present:

```text
decide(state, [A, B, C]) → score(A), score(B), score(C)
```

That ranking does not establish that the menu is complete. **Could a better
action be missing?**

A foundation-model evaluator is valuable precisely because its training carries
knowledge beyond the person who wrote the menu. If it can judge the consequences
of A, B, and C well enough for software to act, it should not be structurally
forbidden from recognizing that none is enough—or that D belongs in the decision.

System One+ adds the missing capabilities: quickly flag when the menu may be
incomplete and, when needed, surface candidate actions worth adding.

## Minimal contract

System One+ adds two capabilities to a System One decision model:

1. **Detect what may be missing.** Flag when the supplied menu may omit a better
   action.
2. **Surface what should be added.** When needed, propose candidate actions outside
   the supplied menu.

That is the definition. Speed, training objectives, gating, calibration, proposal
admission, and model architecture remain open implementation and research choices.

The current prototype explores the contract through three operating modes:

| Mode | Behavior |
| --- | --- |
| `choose` | Rank supplied actions; never generate. |
| `challenge` | Rank, estimate menu adequacy, and selectively propose and score repairs. |
| `suggest` | Generate without a supplied menu, then independently score the proposals. |

## Reference architecture

The implementation in [`ml_parallel/`](ml_parallel/) uses one Qwen backbone and
one shared LoRA adapter. It computes a state prefix once, branches into isolated
questions, and branches again into independently scored candidates.

```text
state x ── shared prefix
  │
  ├── question q₁ ── candidate A ── utility u(x, q₁, A)
  │               ├─ candidate B ── utility u(x, q₁, B)
  │               ├─ candidate C ── utility u(x, q₁, C)
  │               └─ set-attention omission head ── p_missing
  │
  └── question q₂ ── isolated candidate set ── utilities + p_missing

adequate menu  ──> typed ranking; no proposal decoding
triggered menu ──> bounded proposal branch
                  └─> independent scoring ──> admission rule
```

The important properties are:

- **Shared work:** a request can reuse one state prefix across several questions.
- **Candidate isolation:** candidate branches cannot read sibling candidates while
  their utilities are computed.
- **Set-level omission:** a context-conditioned set-attention head evaluates the
  adequacy of the menu, rather than treating low winner confidence as omission.
- **Adaptive compute:** proposal decoding is reserved for requested or triggered
  questions. A typed omission flag can return without generation.
- **Consistent admission:** generated candidates are scored independently. The
  system can use a configured utility margin or a separately fitted admission
  calibration.
- **Joint objectives:** the training target is
  `L = L_score + λ_m L_omission + λ_g L_generation`.

Execution is staged over state, question, and candidate continuations. CUDA uses
explicit FlashAttention kernels; CPU and MPS use a reference implementation. See
[`ml_parallel/DESIGN.md`](ml_parallel/DESIGN.md) for the attention, cache,
training, proposal, and admission contracts.

## Prototype status

The implementation in [`ml_parallel/`](ml_parallel/) includes the model, training
pipeline, service, and data tools.

The included dataset drives the complete training and evaluation pipeline. The next
phase scales the experiment: train new checkpoints, measure ranking and omission
quality, test proposal admission, and benchmark the common and exception paths.

## Run the ML prototype

Requirements: Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cd ml_parallel
uv sync --locked --extra serve
uv run ruff format --check .
uv run ruff check .
uv run sop-parallel-data validate data/smoke-v1
uv run sop-parallel-train --config configs/integrated.toml --dry-run
```

Start the service from a checkpoint:

```bash
uv run sop-parallel-serve --checkpoint runs/parallel-smoke/checkpoint --port 8010
```

CUDA execution has additional pinned requirements. Read the
[`ml_parallel` setup guide](ml_parallel/README.md) before training or serving on a
GPU.

## Open research

System One+ opens a concrete research program:

- datasets where menus are complete, subtly incomplete, redundant, or adversarial;
- metrics that separate ranking quality, omission detection, proposal quality, and
  admission quality;
- calibrated intervention policies with asymmetric costs;
- better set representations and omission objectives;
- bounded proposal mechanisms that surface useful actions without drifting into
  chat;
- real hardware measurements of the common and exception paths;
- failure analysis in agent tools, infrastructure, fraud operations, safety, and
  other consequential decision loops.

If this challenge interests you, work on System One+. Reproduce the prototype,
break its assumptions, train new checkpoints, publish results, propose a better
architecture, or open an issue or pull request with evidence. The point of this
repository is to make the question concrete enough to test.

## License

The repository is licensed under the [Apache License 2.0](LICENSE). Third-party
media attribution and licensing information is listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
