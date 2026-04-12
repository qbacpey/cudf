## Plan: FL64-R4 bw0..64 Generation Strategy

Deliver a planning packet plus architecture and execution runbook for test-only native64 bw0..64 expansion, with strict ownership, parity coverage, and immediate rollback on any gate violation. Canonical strategy location is developer guide docs, with pointer retained in `.github/prompts`.

### Current Understanding Snapshot

**Goal summary**
- Scale test-only native64 prototype logic from bw37 to bw0..64 with deterministic generation.
- Preserve production safety by enforcing denylist on reader/writer/header/runtime paths.
- Define explicit ownership, parity matrix gates, and evidence outputs.
- Prepare staged micro-runs (7+) with confirmation gate after each run when execution starts.

**Confirmed constraints**
- Delivery scope: strategy packet + architecture + execution runbook.
- Canonical strategy path: `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`.
- Ownership strictness: named primary + named backup required per artifact row.
- Generation policy: Option A mandatory; Option B only with explicit blocker waiver.
- Source of truth for generation: JSON bw spec + pinned templates + deterministic generator script.
- Coexistence policy: keep existing bw37 test intact; add generated suite in parallel.
- No-regression gates: split32/raw unchanged, bw37 parity stability guard, target composition unchanged unless approved, denylist enforced.
- Verification strictness: includes full sweep, stability repeats, and full `ctest --output-on-failure` checkpoint at M3.
- Evidence detail: balanced per-bw summary with failed-cell detail + metadata.
- Rollback policy: immediate rollback on first gate violation.
- Evidence report path strictness: `parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json`.
- Run-tag convention: `rt_YYYYMMDD_<letter>`.
- Owner mapping input currently provided: `qbacpey` for all roles (pending explicit confirm that this is intentional).

**Confirmed repo baseline**
- Existing bw37 test suite: `cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu`.
- Existing bw37 CUDA helpers include: `cpp/include/cudf/fastlanes/parquet_fastlanes_native64_bw37_cuda_kernels.inl`.
- Existing strategy packet exists in prompt area: `.github/prompts/fastlanes_native64_r4_generation_strategy.md`.
- `PARQUET_FASTLANES_TEST` wiring exists in `cpp/tests/CMakeLists.txt` and includes bw37 test plus fastlanes pack/unpack sources.
- Legacy generation tree has been removed; retained runner now lives at `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py`.

### Dependency Graph

- FL64-R4-M0 -> FL64-R4-M1 -> FL64-R4-M2 -> FL64-R4-M3 -> FL64-R4-M4 -> FL64-R4-M5 -> FL64-R4-M6 -> FL64-R4-M7
- Parallel note: Within M2, template drafting and schema lint checks can run in parallel once M1 schema is frozen.
- Parallel note: Within M5, report schema draft and path-contract checks can run in parallel once M4 matrix layout is frozen.

### Provisional Micro-Runs (Detailed)

#### 1) FL64-R4-M0: Scope Lock + R3 Evidence Anchor

1. Run ID + title
- `FL64-R4-M0` — Scope Lock + R3 Evidence Anchor

2. Goal
- Freeze include/exclude scope and explicitly reference R3 predecessor evidence locations before any generation design detail.

3. Dependencies
- None.

4. Editable scope (allowlist)
- `docs/cudf/source/developer_guide/index.md` (planning references only)
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md` (if created in execution)
- `.github/prompts/fastlanes_native64_r4_generation_strategy.md` (pointer/reference only)

5. Forbidden scope (denylist)
- `cpp/src/io/parquet/**`
- `cpp/include/cudf/fastlanes/**`
- `cpp/src/fastlanes/**` (except existing test-target linkage remains untouched)
- Any production writer/reader/header/runtime path

6. Concrete edits
- Document baseline constraints and link R3 evidence anchors (bw37 test suite + existing fastlanes reports) in strategy packet preface.
- Add explicit scope denylist section and rollback trigger definition.

7. Validation commands
- Static path audit (`rg -n` preferred): verify no planned editable path intersects denylist.
- If `rg` unavailable: `grep -nE` fallback.

8. Acceptance criteria
- Scope lock section exists and is unambiguous.
- R3 evidence anchors are listed and traceable.

9. Rollback trigger
- Any denylist path appears in proposed edit scope.

10. Evidence to collect
- Scope lock checklist block in strategy packet.
- R3 anchor list with file references.

---

#### 2) FL64-R4-M1: Native64 bw Spec Contract (0..64)

1. Run ID + title
- `FL64-R4-M1` — Spec Table Contract Freeze

2. Goal
- Define canonical schema for native64 bw metadata table and deterministic generation inputs.

3. Dependencies
- Depends on M0.

4. Editable scope (allowlist)
- `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py` (embedded matrix defaults)
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`

5. Forbidden scope (denylist)
- Production fastlanes/parquet source and headers.

6. Concrete edits
- Specify schema fields: `bw`, `words_per_lane`, `words_per_vector`, cross-boundary metadata.
- Define deterministic ordering and schema version key.
- Record ownership row for spec artifact with named primary+backup.

7. Validation commands
- JSON schema lint/validation command (to be specified at execution time).
- Static contract checks in generator design doc.

8. Acceptance criteria
- Table contract covers all bw values 0..64 inclusive.
- Ownership fields contain named primary and backup.

9. Rollback trigger
- Missing any bw entry or ambiguous ownership field.

10. Evidence to collect
- Frozen schema excerpt in strategy packet.
- Machine-readable spec checksum entry.

---

#### 3) FL64-R4-M2: Generator Architecture + Template Determinism

1. Run ID + title
- `FL64-R4-M2` — Generator & Template Architecture

2. Goal
- Define deterministic generator architecture that emits test-only CUDA kernels include from M1 spec.

3. Dependencies
- Depends on M1.

4. Editable scope (allowlist)
- `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py`
- `cpp/include/cudf/fastlanes/native64_cuda_kernels.inl`
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`

5. Forbidden scope (denylist)
- Production encode/decode runtime code paths.

6. Concrete edits
- Define generator input/output contract and deterministic emit rules.
- Include anchor golden snapshot policy for bw 0,1,31,32,33,37,63,64.
- Preserve existing bw37 suite in parallel; do not replace in this milestone.

7. Validation commands
- Determinism check: regenerate and diff must be empty.
- Anchor snapshot consistency check.

8. Acceptance criteria
- Regeneration produces byte-identical output from same inputs.
- Anchor bitwidth generated kernels match template expectations.

9. Rollback trigger
- Non-deterministic generation output or missing anchor snapshot.

10. Evidence to collect
- Generator contract table.
- Snapshot hash list for anchor bitwidth outputs.

---

#### 4) FL64-R4-M3: Generated Harness Glue Design (Parallel bw37 Guard)

1. Run ID + title
- `FL64-R4-M3` — Generated Test Harness Glue Design

2. Goal
- Define generated test harness integration while preserving bw37 test and binary target composition constraints.

3. Dependencies
- Depends on M2.

4. Editable scope (allowlist)
- `cpp/tests/io/parquet_fastlanes_native64_generated_test.cu`
- `cpp/tests/CMakeLists.txt` (only additive test-only wiring if needed and explicitly approved)
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`

5. Forbidden scope (denylist)
- Existing production source/headers.
- Removal or mutation of existing bw37 tests without explicit approval.

6. Concrete edits
- Define fixture naming and filter strategy for anchor/full sweep.
- Define coexistence policy with `ParquetFastLanesNative64Bw37Test.*` guard.
- Keep `PARQUET_FASTLANES_TEST` composition unchanged unless explicitly approved.

7. Validation commands
- Focused: generated anchor tests filter.
- Guard: full `ParquetFastLanesNative64Bw37Test.*`.
- Selected by user: full `PARQUET_FASTLANES_TEST` at M3.

8. Acceptance criteria
- Generated harness path and bw37 guard path are both explicit.
- No unapproved target composition drift.

9. Rollback trigger
- bw37 guard failures or unauthorized target composition changes.

10. Evidence to collect
- Test filter map (anchor/full-sweep/guard).
- Target composition verification snippet.

---

#### 5) FL64-R4-M4: Parity Matrix Contract Freeze (Anchor + Full Sweep)

1. Run ID + title
- `FL64-R4-M4` — Parity Matrix Contract Freeze

2. Goal
- Formalize matrix dimensions, mandatory cells, and pass criteria as binary gates.

3. Dependencies
- Depends on M3.

4. Editable scope (allowlist)
- `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py` (embedded defaults, optional `--config` override)
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`

5. Forbidden scope (denylist)
- Production runtime paths.

6. Concrete edits
- Anchor set: 0,1,31,32,33,37,63,64.
- Full sweep set: 0..64.
- Patterns/bases/counts and stability repeats >=3 for selected seeds.

7. Validation commands
- Matrix config lints and completeness checks.

8. Acceptance criteria
- All required cells explicitly represented with no implicit defaults.
- Pass criteria are binary and machine-checkable.

9. Rollback trigger
- Missing anchor bw or missing full-sweep coverage.

10. Evidence to collect
- Matrix contract checksum.
- Machine-readable completeness report.

---

#### 6) FL64-R4-M5: Executor + Report Schema Design

1. Run ID + title
- `FL64-R4-M5` — Matrix Executor & Report Schema

2. Goal
- Define run orchestration/report schema contract with exact output path enforcement.

3. Dependencies
- Depends on M4.

4. Editable scope (allowlist)
- `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py`
- `parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json`
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`

5. Forbidden scope (denylist)
- Production code paths.

6. Concrete edits
- Enforce output report path format: `parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json`.
- Balanced schema: per-bw summary, failed cell details, run metadata, checksum stability fields.
- Include command-logging requirement in runbook.

7. Validation commands
- Schema validation against sample outputs.
- Path contract check against `<run_tag>` pattern `rt_YYYYMMDD_<letter>`.

8. Acceptance criteria
- Schema validates balanced detail requirements.
- Exact report path policy is enforced and documented.

9. Rollback trigger
- Report path not conformant or schema missing required sections.

10. Evidence to collect
- Sample validated summary JSON.
- Path-conformance check output.

---

#### 7) FL64-R4-M6: Milestone Validation Gates + Rollback Wiring

1. Run ID + title
- `FL64-R4-M6` — Validation Gates & Immediate Rollback Wiring

2. Goal
- Specify execution-time gate sequence and stop-the-line rollback criteria.

3. Dependencies
- Depends on M5.

4. Editable scope (allowlist)
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`
- `.github/prompts/fastlanes_native64_r4_generation_strategy.md` (pointer update only)

5. Forbidden scope (denylist)
- Production files.

6. Concrete edits
- Define mandatory gates:
  - M2 level: generated anchors + bw37 guard.
  - M3 level: anchor matrix + full sweep + stability repeats + bw37 rerun + full `ctest --output-on-failure`.
- Define immediate rollback actions per gate.

7. Validation commands
- Command inventory list with required outputs and pass/fail parsing expectations.
- Search tool policy: `rg -n` preferred, `grep -nE` fallback.

8. Acceptance criteria
- Gate sequence is explicit, ordered, and independently checkable.
- Rollback trigger list is complete and unambiguous.

9. Rollback trigger
- Any required gate failure.

10. Evidence to collect
- Gate checklist with status placeholders.
- Rollback decision log template.

---

#### 8) FL64-R4-M7: Final Packet Consolidation + Handoff Readiness

1. Run ID + title
- `FL64-R4-M7` — Final Strategy Packet Consolidation

2. Goal
- Produce final review-ready planning packet with ownership matrix, unresolved issues, risk matrix, and next-run handoff.

3. Dependencies
- Depends on M6.

4. Editable scope (allowlist)
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md`
- `docs/cudf/source/developer_guide/index.md`
- `.github/prompts/fastlanes_native64_r4_generation_strategy.md` (pointer to canonical doc)

5. Forbidden scope (denylist)
- Any production implementation paths.

6. Concrete edits
- Consolidate all final sections:
  - Ownership matrix with named primary/backup.
  - Mandatory parity matrix and graduation criteria.
  - Unresolved issues log.
  - Risk matrix and milestone gates.
  - Ready-to-run packet for next run only.

7. Validation commands
- Docs consistency checks (sections present, links valid).
- Optional doc build check when execution phase begins.

8. Acceptance criteria
- Packet is executable as written and binary-gated.
- Includes explicit scope boundaries and rollback triggers.
- Canonical doc is in developer guide and indexed.

9. Rollback trigger
- Missing mandatory section or unresolved ownership ambiguity.

10. Evidence to collect
- Final strategy doc revision reference.
- Milestone handoff checklist for next run.

### Relevant Files

- `cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu` — existing bw37 parity baseline and guard reference.
- `cpp/include/cudf/fastlanes/parquet_fastlanes_native64_bw37_cuda_kernels.inl` — existing test-only CUDA helper pattern.
- `cpp/tests/CMakeLists.txt` — current `PARQUET_FASTLANES_TEST` wiring and composition guard.
- `.github/prompts/fastlanes_native64_r4_generation_strategy.md` — current strategy source to keep as pointer/reference.
- `docs/cudf/source/developer_guide/index.md` — canonical strategy doc discoverability in toctree.
- `cpp/examples/parquet_io/docs/fastlanes/fastlanes_native64_r4_generation_strategy.md` — canonical R4 strategy destination.
- `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py` — embedded bw0..64 matrix contract source.
- `cpp/include/cudf/fastlanes/native64_cuda_kernels.inl` — finalized generated kernel include output.
- `cpp/tests/io/parquet_fastlanes_native64_generated_test.cu` — planned generated parity harness glue.
- `cpp/examples/parquet_io/tools/tests/run_parity_matrix.py` — planned parity matrix executor.

### Verification (Plan-level)

1. Scope verification: ensure all planned editable paths are test/docs only and do not intersect denylist.
2. Ownership verification: each artifact row has named primary + named backup.
3. Matrix verification: anchor set and full sweep sets are explicit and complete.
4. Gate verification: M2 and M3 validation gates are fully specified with binary pass/fail criteria.
5. Evidence verification: report path uses strict convention and run-tag format.

### Decisions Captured

- Canonical strategy location: developer guide docs, prompt file as pointer.
- Generation approach: deterministic scripted generation only (fallback by explicit blocker waiver).
- Compatibility strategy: keep bw37 suite intact in parallel during R4.
- Validation strictness: includes full ctest checkpoint at M3 and immediate rollback on violation.
- Execution cadence (future): mandatory confirmation after each micro-run.

### Open Questions (Narrow)

- None currently.

### Additional Decisions Captured

- Owner mapping: `qbacpey` is intentionally primary and backup for all owner/reviewer roles in this R4 packet.
- M7 includes a post-R4 deprecation criteria note for the bw37-only suite.
