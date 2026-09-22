# Parallel-question System One+: staged execution

One Qwen/LoRA model learns candidate utility, set-level omission detection, and
proposal generation. The model semantics and three objectives are unchanged.
Execution now follows the dependency tree instead of packing every token under a
request-wide dense attention mask.

## State, question, and candidate stages

```text
state → one shared KV prefix
  ├── question 1 → state/question prefix → candidate rows A, B, C
  ├── question 2 → state/question prefix → candidate rows X, Y
  └── question 3 → state/question prefix → ...
```

1. Run the state once and retain each layer's K/V tensors.
2. Batch question continuations against that state prefix.
3. Batch candidate continuations against their corresponding state/question prefixes.
4. Read each candidate endpoint for utility; read each question's context and candidate
   endpoints through the existing set-attention omission head.
5. For supervised proposal targets, batch proposal continuations against each question's
   assembled menu cache. At inference, only requested/triggered questions decode proposals.

Each stage groups rows by prefix and continuation lengths, then splits buckets
at `branch_batch_size` (default 16). `branch_bucket_width=0` retains exact-length
batches. A positive width groups nearby lengths with right padding. CUDA attention
unpads Q/K/V and passes per-row lengths to FlashAttention; the portable reference
uses a local rectangular boolean mask. Padding is excluded from attention and persistent caches;
real token position IDs are unchanged. Padded batches are optional pending hardware
measurements, not a claimed speed improvement. Questions in different buckets may
require different batches. This is several
staged backbone calls, not a single forward call. Across the initial stages, each
state, question, and candidate token is processed once.

An empty menu still has a question/context representation and can generate suggestions.
Its omission target remains undefined. Candidate utilities remain independent of
siblings. The scorer still requires self-contained action descriptions.

## Attention and positions

`staged.py` registers `sop_staged_sdpa` through Transformers' attention interface.
Qwen retains its own embeddings, projections, RoPE, normalization, MLPs, and LoRA.
`attention.py` implements the CUDA and reference operations:

- State and branch continuations use `flash_attn_varlen_func`, with unpadded tokens,
  cumulative query/key lengths, native grouped-query attention, and causal alignment.
- Proposal decoding uses `flash_attn_with_kvcache` with preallocated storage and
  device-side lengths. Qwen applies RoPE once, before inserting keys; FlashAttention
  receives no rotary parameters and does not apply it again.
- CPU/MPS and explicitly selected CUDA reference mode use PyTorch SDPA. A multi-token
  continuation uses `causal_lower_right(query_length, key_length)`; a one-token
  continuation reads its entire prefix and current token.

Lower-right alignment matters because a continuation has fewer query tokens than
key tokens. Passing the usual upper-left causal flag for a non-square matrix would
incorrectly hide part of the prefix. An independent evaluation should compare both
forward outputs and backward gradients against explicit causal matrices.

The default CUDA policy requires FlashAttention and BF16. FP16 is also available for
inference. Unsupported hardware, FP32 Flash inputs, missing dependencies, and unexpected
mask inputs fail clearly instead of silently selecting quadratic attention. CUDA
reference execution must be explicitly selected. CPU/MPS use FP32 reference execution.
There is no global square mask or padded attention matrix on the CUDA Flash path.
Model loading selects the frozen backbone dtype; the LoRA parameters and decision
heads remain FP32, while backbone and vocabulary operations use the selected autocast.
Sliding-window configurations and nonzero attention dropout are rejected, because
the execution equivalence here assumes full causal attention and deterministic
backbone attention. The pinned Qwen configurations meet these requirements.

Question positions start at the state length. Each candidate starts at its own
state/question prefix length. Proposal positions start after that question's longest
candidate branch, as before. Physical KV lengths can therefore differ from logical
RoPE positions; both are handled explicitly.

## Shared immutable cache segments

A `KVSegment` stores only a node's own per-layer K/V tensors. A `BranchCache` is an
immutable tuple of segment references. Question caches reference the same state
segment; candidate caches reference their question's state and question segments.
The persistent caches do not repeat the state tensors for every branch.

For a kernel call, an ephemeral cache collector assembles the appropriate contiguous
K/V batch and captures only the newly computed suffix tensors. This assembly can
create temporary copies; it is not a zero-copy paged-attention implementation.
The row cap bounds each assembly. No collector mutates a parent segment.

A question's proposal context references:

```text
state + own question + candidate A suffix + candidate B suffix + ...
```

The state/question prefix appears once, and no other question is included. Candidate
K/V tensors were computed independently and retain their original RoPE positions.
Inference proposal decoding materializes these references **once per search row**
into `DecodeCache`, a preallocated inference-only store. Each new token writes to
its next physical slot; no growing sequence of segments or per-token full-prefix
concatenation is used. Logical RoPE positions are still supplied separately.
Triggered questions and independent sampled searches decode together in batches.
Each row owns its device-side grammar, budget, and RNG stream. Structural masking,
finite-logit checks, greedy selection, and sampled selection do not read tensors
back into Python. Seeded sampling generates device-side uniforms and searches the
batched cumulative token distribution; vocabulary scores never move to the CPU.
An ordinary-token/content table is prepared once per tokenizer, outside decoding.
Final parsing validates full decoded text, including multi-token Unicode whitespace.

Every `decode_check_interval` steps (default 8, range 1–32), the host reads one small
status vector. Finished token histories are then transferred and decoded. Until
that check, finished rows run dummy steps with frozen status and cache length;
their results cannot affect active rows. Completed or failed rows are removed at
the check; compaction copies surviving cache rows only when the active set changes.
This trades up to `interval - 1` dummy steps for fewer host synchronizations. It
does not eliminate Python kernel launches or periodic batch completion checks.
`branch_batch_size` caps rows, and `max_decode_batch_tokens` bounds the padded
allocation (`rows * maximum capacity`), so large groups are processed in chunks.
The original assessment segments are never mutated.
New-candidate scoring references only the original state/question segments and
batches the new candidate continuations. Existing utilities remain unchanged.

## Joint training and checkpointing

`L = L_score + lambda_m L_omission + lambda_g L_generation`.

Every training prefix tensor stays attached to autograd. Sharing a segment means
all child losses contribute gradients to the same prefix computation and shared
LoRA adapter. No training prefix is detached or computed under `no_grad`.

`gradient_checkpointing` uses non-reentrant checkpointing around complete continuation
batches. Unique prefix tensors are explicit checkpoint inputs, and newly computed
K/V tensors are outputs. Each execution/recomputation constructs a fresh collector.
This avoids mutable-cache replay and Hugging Face's inference-cache/checkpointing
interaction. Gradient equivalence with and without checkpointing, including shared
prefix tensors, remains a required validation target.

The trainer groups same-state examples within the existing optimizer accumulation
batch. Each objective retains normalization by its active example count across that
step. Neither objectives nor label semantics change. The grouping limit remains
`max_questions_per_pass`; it describes questions per shared-state group, not one
backbone invocation. The report's `training_shared_state_groups` reflects this.

## Limits, checkpoints, and verification

Limits have separate meanings:

- `max_length`: logical branch positions, including the longest candidate and proposal
  continuation. Also bounded by the model's position capacity.
- `max_packed_tokens`: total unique input-token work per request/training group.
- `max_cache_tokens`: per-question proposal cache capacity, including **all** candidate
  suffixes and generated tokens. Training proposal targets respect this capacity too.
- `max_decode_batch_tokens`: total padded cache slots in an active decoding batch.

Wide menus no longer fail merely because the sum of independent candidate lengths
exceeds the logical position limit. Generation still consumes a cache containing the
whole menu; it cannot use a longest-branch-only memory budget. A closed gate does not
reserve generation capacity. Exhausted proposal budgets retain the supplied ranking.

Checkpoints keep architecture `parallel_utility_set_attention_v1` and encoding
version 2, with checkpoint version 4, execution version `staged_flash_v3`, the retained
registered backend name `sop_staged_sdpa`, and precision/batching/budget settings.
Version-1, version-2, and version-3 parallel checkpoints
can load the same weights into staged execution. The loader records their original
execution in `loaded_from_execution` and leaves the original checkpoint unchanged.
Other architectures remain rejected. The base model, shared adapter, utility head,
omission modules, and training losses are unchanged. Older checkpoints receive unfitted
admission-calibration buffers because the sampling/grammar implementation changed.
Ranking temperature is not used for admission. PyTorch 2.8.0, Transformers 4.57.6,
PEFT 0.21.0, and the optional Linux CUDA dependency FlashAttention 2.8.3 are exactly pinned.

Production encoding contains token records and boundaries, not dense mask builders.
Before relying on the prototype, independently verify:

- Outputs, proposal logits, and each objective's gradients against the dense reference.
- Checkpointed/uncheckpointed training and gradients through shared prefixes.
- Question/candidate order, isolation, and prevention of teacher-target leakage.
- Cache values, immutable shared storage, and subsequent generation/new scoring.
- Batch-size independence, lower-right causal alignment, and actual row batching.
- Save/reload/continuation, HTTP validation, and per-question failure handling.

This release includes no test suite or benchmark evidence for those properties.
CUDA kernels, cached decoding, grammar constraints, latency, throughput, and VRAM
behavior all require independent validation on appropriate hardware.

## Proposal searches and admission calibration

The default remains one grammar-constrained greedy continuation producing up to
`n_generate` actions. `proposal_branches > 1` requires `temperature > 0` and at least
one action of budget per branch. The total action budget is divided among branches;
outputs are combined and exact duplicates removed, so fewer actions may be returned.
A seed is hashed with the stable request question ID and proposal branch index using
SHA-256 over a structured tuple. Row/owner indices are never part of seed derivation.
Each question/branch has its own RNG stream, preserved under question reordering,
gate filtering, batch splitting, and compaction. The single-question default ID is
`question`; batched generation requires explicit unique IDs.
Per-search errors remain visible even if another search succeeds. Generation
failures are not silently treated as empty successful proposals.

The ranking temperature only normalizes choice utilities. Admission uses its own
positive scale and bias fitted against the binary label “strictly better than the
model-selected supplied incumbent.” Equal, redundant, or worse actions are negative.
The fitted probability is `sigmoid(admission_scale * utility_gap + admission_bias)`.
An admitted action must additionally have a positive raw utility gap.

`admission.py` collects actual outputs on frozen development scenarios, with the
incumbent, exact checkpoint fingerprint, and generation policy attached. Human
review supplies the binary outcomes. Fitting rejects pending labels, other splits,
stale checkpoints, changed policies, duplicate records, and incorrect incumbents.
Optional validation reviews must use disjoint scenario groups. In-sample fit metrics
and held-out validation metrics are reported separately; fitting is not validation.

Serving only uses this calibration when the active generation policy and budgets
match those recorded at fitting, including the decoder, attention backend, and compute
precision. Otherwise it uses the explicitly configured raw
`admission_margin` and returns `admission_probability: null` (also `preference: null`).
The response identifies the admission rule and calibration status. Continuing joint
training invalidates and clears the old admission fit. No reviewed research data or
calibrated production probabilities are supplied by this prototype.
