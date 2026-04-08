## Plan: FastLanes64 Applicability bw37-first

Validate that native64 GPU decoding is feasible using a low-risk, test-first path: prove correctness on bw37 first, preserve existing split64/raw behavior, then produce a concrete generation strategy for bw0..64 and a ready integration blueprint.

### Current Understanding Snapshot

Confirmed facts
- CPU reference fastlanes pack/unpack already supports bw0..64 for 64-bit words.
- GPU unpack dispatch is currently bw0..32 only, and INT64 decode is hard-wired to split32 low/high streams.
- Header validation in current fastlanes metadata helpers enforces component bitwidth in range 1..32.
- Host-side page gather for INT64/UINT64 and CPU-side int64 fastlanes encoding are already present; current int64 encode path uses split32 and includes explicit native64 TODO markers.
- Writer/requested encoding surface currently exposes RAW and SPLIT64 only.

Inferences
- Applicability is strong, but first success criterion should be a test-only native64 decoder prototype to avoid early regressions in production decode routing.
- A new explicit encoding value is the cleanest long-term routing strategy for native64 integration.

Open decisions
- None blocking for applicability stage (scope and verification depth selected).
- For later integration stage, exact native64 header contract (reserved flag usage vs extended layout) still requires final decision at implementation-time design review.

### Decisions Captured From User Alignment
- Scope now: Applicability plus bw37 prototype only.
- Must-have outcomes now: applicability verdict with blockers/risks, bw37 GPU-vs-CPU parity gate, generation strategy for bw0..64, integration plan for later native64 writer/reader work.
- Non-regression minimum: keep existing SPLIT64 behavior unchanged and keep existing RAW INT32 behavior unchanged.
- Out of scope for early runs: performance optimization, nested/repeated support, non-INT64/UINT64 expansion.
- Prototype placement: test-only kernel path first (no production routing changes).
- Future native64 signal: new parquet encoding enum FASTLANE_BITPACK_NATIVE64.
- API timing: expose new public enum only after bw0..64 parity is proven.
- Validation preference: full libcudf tests each run; broader checkpoint required after bw37 parity.
- Delivery preference: very small runs, run IDs FL64-R1/FL64-R2 style, detailed evidence report each run.

### Dependency Graph
- FL64-R1 -> FL64-R2 -> FL64-R3 -> FL64-R4
- FL64-R5 depends on FL64-R3 and can proceed in parallel with FL64-R4.

### Micro-runs (Provisional)

1. FL64-R1: CPU-oracle harness and bw37 fixture setup
- Goal: establish deterministic CPU oracle path and fixture generation for bw37 and boundary widths without touching production decode routing.
- Dependencies: none.
- Editable scope (allowlist):
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp or a new focused fastlanes native64 test file under /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/
  - Build registration files only if required for new test compilation.
- Forbidden scope (denylist):
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/io/parquet_schema.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/io/types.hpp
- Concrete edits:
  - Add test helpers that generate CPU-packed payloads using existing generated pack reference.
  - Add deterministic/adversarial datasets for bw37 and required boundary widths list.
- Validation approach:
  - Full libcudf tests per user preference, plus targeted fastlanes test filter for quicker signal.
- Acceptance criteria:
  - Tests compile and produce deterministic CPU oracle fixtures.
  - No behavior change in existing split64/raw tests.
- Rollback trigger:
  - Any failure in existing fastlanes split64/raw tests or inability to deterministically reproduce fixture payload.
- Evidence to collect:
  - Touched files, test command list, pass/fail summary, fixture checksum snippets.

2. FL64-R2: bw37 GPU native64 decode test-kernel prototype
- Goal: add a test-only GPU kernel for native64 bw37 decode and compare outputs against CPU oracle.
- Dependencies: FL64-R1.
- Editable scope (allowlist):
  - Test-only CUDA helper/test files in /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/
  - Minimal helper header in test tree if needed.
- Forbidden scope (denylist):
  - Production decode routing files and page header contracts:
    - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
    - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh
    - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
- Concrete edits:
  - Implement a dedicated bw37 unpack/decode kernel in test scope only.
  - Add bitwise parity assertion GPU decode == CPU decode for INT64 and UINT64 views.
- Validation approach:
  - Full libcudf tests plus targeted native64 prototype test filter.
- Acceptance criteria:
  - bw37 parity passes on randomized and adversarial sets.
  - Existing split64/raw paths remain unchanged and passing.
- Rollback trigger:
  - Any existing regression or any non-bitwise-equal mismatch in parity checks.
- Evidence to collect:
  - Mismatch counters (expected zero), seed list, test logs, file diff summary.

3. FL64-R3: bw37 robustness and checkpoint gate
- Goal: harden bw37 proof with boundary-adjacent cases and complete checkpoint validation before broader generation strategy.
- Dependencies: FL64-R2.
- Editable scope (allowlist):
  - Same test-only scope as FL64-R2.
- Forbidden scope (denylist):
  - Same production denylist as FL64-R2.
- Concrete edits:
  - Add stress variants (tail sizes around 1024 boundaries, extreme signed values, pathological bit patterns) for bw37 path.
- Validation approach:
  - Full libcudf tests (mandatory).
  - Milestone checkpoint after pass.
- Acceptance criteria:
  - Stable pass across repeated runs with fixed seeds.
  - No new flakes in fastlanes-related suites.
- Rollback trigger:
  - Flaky results or failures outside touched test scope indicating hidden coupling.
- Evidence to collect:
  - Repeated-run stability table, failing seed reproduction recipe if any, checkpoint summary.

4. FL64-R4: bw0..64 generation strategy packet
- Goal: produce implementable strategy for scaling prototype logic to bw0..64 with explicit risk controls.
- Dependencies: FL64-R3.
- Editable scope (allowlist):
  - Planning/design docs only (or test-only generation utility if explicitly desired).
- Forbidden scope (denylist):
  - Production reader/writer/runtime files.
- Concrete edits:
  - Define generation approach options: scripted generation vs assisted manual generation.
  - Define mandatory parity matrix: 0,1,31,32,33,37,63,64 then full sweep 0..64.
  - Define criteria for graduating from test-only to production path.
- Validation approach:
  - Validate strategy completeness against dependency map and file ownership.
- Acceptance criteria:
  - Clear, executable matrix and ownership for each generated artifact.
- Rollback trigger:
  - Ambiguous ownership or missing parity coverage.
- Evidence to collect:
  - Strategy document, matrix table, unresolved issue list.

5. FL64-R5: native64 integration blueprint (design only in this stage)
- Goal: deliver dependency-safe blueprint for later integration into enum/header/writer/reader routing.
- Dependencies: FL64-R3 (proof complete).
- Parallelism: can run in parallel with FL64-R4.
- Editable scope (allowlist):
  - Planning document only in current stage.
- Forbidden scope (denylist):
  - Production edits in this stage.
- Concrete edits:
  - Specify exact future file touchpoints and ordering for enum additions, kernel masks, header validation, decode routing, and writer selection.
  - Preserve split64/raw fallback until full native64 parity is proven.
- Validation approach:
  - Design review against current constraints and regression suite mapping.
- Acceptance criteria:
  - No unresolved dependency cycles; each future run is reversible and narrowly scoped.
- Rollback trigger:
  - Any design choice that forces mixed concerns in one run.
- Evidence to collect:
  - Integration dependency map, run-by-run allowlist/denylist draft, risk matrix.

### Relevant files (high-priority anchors)
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/pack.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/unpack.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fls_gen/unpack/unpack.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/io/types.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/io/parquet_schema.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl_preprocess_utils.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_writer_test.cpp

### Verification Execution Policy Packet (for later implementation agent runs)
1. Use existing terminal and verify remote context first with hostname and whoami.
2. If not remote, connect using ssh qchen@fng01.lab.tuda.systems.
3. Initialize env in order: source conda.sh, conda activate cudf_dev, export CPATH.
4. TARGET_BRANCH currently maps to fastlane-working so CUDF_HOME is /home/qchen/04_GPUFileFormat-cudf/cudf-fastlane.
5. Always build from CUDF_HOME via build.sh only.
6. If /tmp pressure occurs, set TMPDIR, create directory, set PARALLEL_LEVEL=4, rerun build.sh.
7. Prefer rg -n for verification scans; fallback grep -nE if rg unavailable.
8. If writer tests are needed in later runs, resolve writer gtest binary with PARQUET_WRITER_TEST first, fallback PARQUET_TEST.
9. Keep large artifacts remote; pull small reports via sync_parquet_io_shared_small.sh pull.
10. Report exact commands, final input path, output/report/artifact paths, archive status, search tool used, and resolved writer gtest binary path when applicable.

### Scope boundaries
Included now
- Applicability verification and bw37 prototype planning path.
- Generation strategy for full bitwidth coverage.
- Integration blueprint only (no production integration execution in this stage).

Excluded now
- Performance tuning.
- Nested/repeated support.
- Non-INT64/UINT64 expansion.
- Production-path native64 integration code edits.

### Plan changes since last iteration
- Initial version created from discovery plus four rounds of user alignment questions.
- Scope narrowed to test-only bw37 prototype first; integration work remains blueprint-only for this stage.

### Still unresolved
- Later-stage native64 header contract detail (exact field-level representation) to be finalized during integration-design phase.
