<!-- markdownlint-disable MD007 MD022 MD029 MD032 MD058 MD060 -->

# FL64-R4: bw0..64 Generation Strategy Packet

This document is the canonical source of truth for FL64-R4 planning and staged implementation.
The prompt copy at `.github/prompts/fastlanes_native64_r4_generation_strategy.md` is retained for workflow convenience and must point back to this document.

## Goal
Produce an implementable and low-risk strategy to scale the FL64 prototype logic from bw37 to bw0..64 with explicit ownership, parity coverage, and rollback controls.

## Dependency
- Required predecessor: FL64-R3 (bw37 GPU parity and stability complete).

## FL64-R3 Evidence Anchors
- Test baseline: `cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu`
- Test CUDA helper include: `cpp/tests/io/parquet_fastlanes_native64_bw37_cuda_kernels.inl`
- Fastlanes gtest target wiring: `cpp/tests/CMakeLists.txt` (`PARQUET_FASTLANES_TEST`)
- Prior fastlanes integration report: `cpp/examples/parquet_io/docs/fastlanes/FASTLANES_INTEGRATION_REPORT_2026-03-28.md`
- Prior split64 sensitivity report: `cpp/examples/parquet_io/docs/fastlanes/INT64_SNAPPY_DELTA_VS_FASTLANES_REPORT_2026-04-02.md`

Reference validation command for predecessor evidence:

```bash
cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64Bw37Test.*
```

## Scope Guardrails
- Allowed scope:
  - Planning and design documents.
  - Test-only generation utility and test-only helper artifacts.
- Forbidden scope:
  - Production reader, writer, header/runtime paths.
  - `cpp/src/io/parquet/**`
  - `cpp/include/cudf/fastlanes/**`
  - `cpp/src/fastlanes/**` (except existing test-target linkage remains untouched)

## FL64-R4-M0 Scope Lock Checklist
- [x] Canonical strategy packet exists under `docs/cudf/source/developer_guide/`.
- [x] FL64-R3 evidence anchors are explicitly listed in this packet.
- [x] Scope denylist is explicit and path-based.
- [x] Rollback trigger is binary and immediate.
- [x] No production-path files are modified by FL64-R4-M0.

## Ownership and Artifact Matrix
| Artifact | Type | Owner | Backup Reviewer | Source of Truth | Output Path | Gate |
|---|---|---|---|---|---|---|
| Native64 bw spec table (bw, words_per_lane, cross-boundary map) | Generated metadata (test-only) | qbacpey | qbacpey | Script input spec | cpp/tests/io/fastlanes_native64_gen/spec/native64_bw_table.json | FL64-R4-M1 |
| CUDA kernel include (bw0..64, test-only) | Generated code | qbacpey | qbacpey | Generator + golden snapshots | cpp/tests/io/fastlanes_native64_gen/generated/native64_bw_kernels.inl | FL64-R4-M2 |
| CPU oracle compare harness glue | Handwritten test glue | qbacpey | qbacpey | Test source | cpp/tests/io/parquet_fastlanes_native64_generated_test.cu | FL64-R4-M2 |
| Parity matrix executor (anchor + full sweep) | Test utility | qbacpey | qbacpey | Matrix config | cpp/tests/io/fastlanes_native64_gen/tools/run_parity_matrix.py | FL64-R4-M3 |
| Run evidence summary | Report | qbacpey | qbacpey | CI/remote logs | parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json | FL64-R4-M3 |
| Unresolved issues log | Design note | qbacpey | qbacpey | This packet + run feedback | docs/cudf/source/developer_guide/fastlanes_native64_r4_generation_strategy.md | FL64-R4-M4 |

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
Rollback R4 outputs immediately if either condition occurs:
- Ownership is ambiguous for any generated artifact.
- Parity coverage is missing any required anchor bitwidth or any bw in 0..64 sweep.

## Evidence to Collect
- Strategy document (this file).
- Matrix execution summary report per run tag.
- Unresolved issue list updates.

## Unresolved Issues
- Native64 parquet header contract detail (encoding metadata representation) remains pending and is intentionally deferred to post-R4 production design review.

## Post-R4 bw37 Deprecation Criteria (Deferred)
The bw37-only suite may be considered for deprecation only after all of the following are true:
1. Generated bw0..64 anchor and full-sweep parity gates are passing and stable.
2. Generated harness has equivalent or stronger failure diagnostics than bw37-only tests.
3. A dedicated transition review confirms no observability regression for bw37 failure triage.
4. Removal proposal is approved in a separate post-R4 change.

## Risk Matrix
| Risk | Severity | Detection | Mitigation |
|---|---|---|---|
| Generator emits incorrect bw mapping broadly | High | Anchor snapshot drift, parity mismatch spikes | Deterministic regeneration checks and anchor snapshot gate |
| Coverage gap in anchor/full sweep | High | Matrix config completeness check, missing-cell report | Binary rollback trigger on any missing required cell |
| Regressions to existing bw37 parity diagnostics | Medium | Dedicated bw37 guard run in every major validation phase | Keep bw37 suite in parallel through R4 and require explicit deprecation review |
| Report path drift outside shared root contract | Medium | Path policy check in matrix summary JSON | Fail report generation when path policy is violated |
| Ownership ambiguity on generated artifacts | Medium | Ownership matrix audit in packet review | Binary rollback trigger on any ambiguous row |

## Milestone Gates
1. FL64-R4-M1 gate:
  - `native64_bw_table.json` exists and deterministically regenerates.
  - Ownership row for spec artifact is explicit.
2. FL64-R4-M2 gate:
  - Generated include deterministically regenerates from spec + template.
  - Anchor snapshot exists for bw 0,1,31,32,33,37,63,64.
3. FL64-R4-M3 gate:
  - Anchor matrix parity passes.
  - Full sweep parity matrix passes.
  - Stability repeats (>=3) pass for selected seeds.
  - `ParquetFastLanesNative64Bw37Test.*` guard passes.
  - Full `ctest --output-on-failure --no-tests=error` checkpoint passes.
4. FL64-R4-M4 gate:
  - Matrix contract files enumerate required anchors, sweep range, scenarios, bases, counts.
5. FL64-R4-M5 gate:
  - `r4_matrix_summary.json` conforms to schema and report path policy.
6. FL64-R4-M6/M7 gate:
  - Strategy packet contains risk matrix, milestone gates, execution order, and ready-to-run packet.

## Execution Order Recommendation
1. M1: Freeze deterministic bw spec table.
2. M2: Freeze deterministic generator/template and generated include output.
3. M3: Add generated parity harness in parallel with existing bw37 suite.
4. M4: Freeze matrix contract (anchors + full sweep + stability).
5. M5: Freeze execution/report tooling and schema.
6. M6: Freeze gate/rollback runbook.
7. M7: Final packet consolidation and handoff.

## Ready-to-run Packet (Next Run Only)

Run ID: FL64-R4-M3 remote validation checkpoint

Goal:
- Validate generated native64 harness against anchor/full-sweep/stability plus bw37 guard.

Dependencies:
- M1 and M2 artifacts generated and synced to remote worktree.

Editable scope (allowlist):
- No additional code edits required for this run.
- Run-only outputs under `${PARQUET_IO_SHARED_ROOT}/reports/cudf-fastlane/<run_tag>/`.

Forbidden scope (denylist):
- No production path edits.
- No writer/reader/header runtime edits.

Validation commands:
1. Build tests via `${CUDF_HOME}/build.sh libcudf tests`.
2. Run targeted gtests:
  - `PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64GeneratedTest.AnchorMatrixParity`
  - `PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64GeneratedTest.FullSweepParityMatrix`
  - `PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64GeneratedTest.StabilityRepeatsSelectedSeeds`
  - `PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64Bw37Test.*`
3. Run full checkpoint: `ctest --output-on-failure --no-tests=error`.
4. Generate summary JSON via `run_parity_matrix.py` at exact report path contract.

Acceptance criteria:
- All commands exit 0.
- Summary JSON indicates `all_passed=true` and `path_policy_ok=true`.

Rollback trigger:
- First failing command or path policy violation triggers immediate rollback of R4 outputs.

Evidence to collect:
- Exact command list and return codes.
- `r4_matrix_summary.json` in shared reports path.
- Note whether `rg -n` or `grep -nE` fallback was used.
