<!-- markdownlint-disable MD022 MD032 MD058 MD060 -->

# FL64-R4: bw0..64 Generation Strategy Packet

> Canonical source: `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`.
> Keep this prompt copy aligned for workflow usage, but apply normative updates to the canonical doc first.

## Goal
Produce an implementable and low-risk strategy to scale the FL64 prototype logic from bw37 to bw0..64 with explicit ownership, parity coverage, and rollback controls.

## Dependency
- Required predecessor: FL64-R3 (bw37 GPU parity and stability complete).

## Scope Guardrails
- Allowed scope:
  - Planning and design documents.
  - Test-only generation utility and test-only helper artifacts.
- Forbidden scope:
  - Production reader, writer, header/runtime paths.

## Ownership and Artifact Matrix
| Artifact | Type | Owner | Backup Reviewer | Source of Truth | Output Path | Gate |
|---|---|---|---|---|---|---|
| Native64 bw matrix contract (bw, words_per_lane, cross-boundary map) | Embedded metadata (test-only) | FastLanes test owner | IO parquet owner | Script defaults in run helper | cpp/examples/parquet_io/tools/tests/run_parity_matrix.py | FL64-R4-M1 |
| CUDA kernel include (bw0..64, test-only) | Generated code | FastLanes test owner | CUDA reviewer | Finalized checked-in include | cpp/include/cudf/fastlanes/native64_cuda_kernels.inl | FL64-R4-M2 |
| CPU oracle compare harness glue | Handwritten test glue | FastLanes test owner | QA owner | Test source | cpp/tests/io/parquet_fastlanes_native64_generated_test.cu | FL64-R4-M2 |
| Parity matrix executor (anchor + full sweep) | Test utility | QA owner | FastLanes test owner | Embedded defaults + CLI overrides | cpp/examples/parquet_io/tools/tests/run_parity_matrix.py | FL64-R4-M3 |
| Run evidence summary | Report | QA owner | IO parquet owner | CI/remote logs | parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json | FL64-R4-M3 |
| Unresolved issues log | Design note | IO parquet owner | FastLanes test owner | This packet + run feedback | cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md | FL64-R4-M4 |

## Generation Approach Options

### Option A: Scripted Generation (preferred)
- Method:
  - Use one canonical bw table (0..64) and generate CUDA test kernels and dispatch stubs.
  - Emit deterministic files from pinned templates.
- Pros:
  - Low drift risk across 65 bitwidths.
  - Repeatable regeneration and easier review diffs.
  - Better ownership clarity via explicit generator inputs.
- Risks:
  - Generator bugs can propagate broadly.
- Controls:
  - Golden snapshots for anchor bitwidths.
  - Generator unit checks for word count and boundary crossings.
  - CI block if generated output differs from committed output.

### Option B: Assisted Manual Generation (fallback)
- Method:
  - Generate anchor bitwidths automatically and hand-write remaining bw code in small batches.
- Pros:
  - Easier debugging for a few edge bitwidths.
- Risks:
  - High maintenance cost and drift risk.
  - Inconsistent ownership and style.
- Controls:
  - Mandatory two-reviewer signoff per batch.
  - Strict parity gate after each batch.

### Decision Rule
- Use Option A unless a blocker is proven that prevents deterministic generation.
- If Option B is used for any bw, record justification and owner in Unresolved Issues.

## Mandatory Parity Matrix

### Phase 1: Anchor Matrix (must pass first)
- Bitwidths: 0, 1, 31, 32, 33, 37, 63, 64.
- For each anchor bw:
  - CPU pack -> GPU decode parity.
  - GPU encode -> CPU unpack parity.
  - GPU encode -> GPU decode roundtrip parity.
  - Signed INT64 bit-cast parity checks.

### Phase 2: Full Sweep Matrix (0..64)
- Sweep all bitwidths 0 through 64.
- Per bw scenarios:
  - randomized, adversarial, pathological patterns.
  - bases: 0, near INT64 min, near INT64 max, and high unsigned base.
  - counts: 1023, 1024, 1025, 2047, 2048, 2049.
- Minimum pass criteria:
  - zero parity mismatches for all matrix cells.
  - stable checksums across repeated runs (>= 3 repeats for selected seeds).

## Graduation Criteria: Test-Only -> Production Candidate
All conditions must be true before any production-path proposal:
1. Anchor matrix passed for all required checks.
2. Full sweep 0..64 passed with zero mismatches.
3. Ownership matrix has no ambiguous owner fields.
4. Generated artifacts are reproducible from committed generator inputs.
5. Existing split32/raw non-regression tests pass unchanged.
6. Header contract proposal for native64 has explicit review approval.

## Validation Approach
Validate strategy completeness with a dependency and ownership checklist:
1. Confirm FL64-R3 evidence exists and links to bw37 parity reports.
2. Confirm each generated artifact row has one primary owner and one reviewer.
3. Confirm anchor matrix is explicitly listed and includes 0,1,31,32,33,37,63,64.
4. Confirm full sweep 0..64 is explicitly required.
5. Confirm graduation criteria and rollback trigger are testable and binary.
6. Confirm no production path files are modified by R4 activities.

## Acceptance Criteria
- Strategy packet exists and is executable as written.
- Matrix and ownership are explicit per artifact.
- Unresolved issues section is present and non-empty if any open items remain.

## Rollback Trigger
Rollback R4 outputs if either condition occurs:
- Ownership is ambiguous for any generated artifact.
- Parity coverage is missing any required anchor bitwidth or any bw in 0..64 sweep.

## Evidence to Collect
- Strategy document (this file).
- Matrix execution summary report per run tag.
- Unresolved issue list updates.

## Unresolved Issues
- Native64 parquet header contract detail (encoding metadata representation) remains pending and is intentionally deferred to post-R4 production design review.
