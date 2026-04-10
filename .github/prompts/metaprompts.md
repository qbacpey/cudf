```text
BEGIN AGENT REMOTE CONTRACT

I am working from a local machine without GPU. You can assume `lsyncd` is running. You must execute GPU build/test on remote server.

Task variables you must read first:
- TARGET_BRANCH: fastlane-working
- INPUT_PATTERN: (example: (base) qchen@fng01:~/04_GPUFileFormat-cudf/large_input$ l
0003-cudf-DELTA.parquet tpch1-SNAPPY-cudf.parquet  tpch1-nocomp-cudf.parquet)
- RUN_TAG: short tag for this run (example: rt_20260404_a)

Execution policy:
1. Use my current terminal whenever possible. Do not open extra terminals unless blocked.
2. First verify whether you are already in SSH environment by checking hostname/user.
3. If not on remote, connect with: ssh `qchen@fng01.lab.tuda.systems`
4. In remote shell, set environment in this order:
   `source ~/miniconda3/etc/profile.d/conda.sh`
   `conda activate cudf_dev`
   `export CPATH="$CONDA_PREFIX/include/rapids:$CONDA_PREFIX/include"`
5. Do not run set -u before conda activate.
6. For validation text search, use `rg -n` when available. If `rg` is unavailable, use
   `grep -nE` with the same pattern and file list.
7. For long remote commands with nested single quotes, prefer
   `cat <<'EOF' | ssh ... bash -s` to avoid quote-break issues.

Branch mapping:
1. If TARGET_BRANCH=main:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/GPUFileFormat-cudf`
3. If TARGET_BRANCH=nvcomp-working:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-nvcomp`
4. If TARGET_BRANCH=fastlane-working:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane`

Roundtrip data roots:
- `export ROUNDTRIP_INPUT_ROOT=/home/qchen/04_GPUFileFormat-cudf/large_input`
- `export PARQUET_IO_SHARED_ROOT=/home/qchen/04_GPUFileFormat-cudf/parquet_io_shared`

Build rule:
1. Always build from ${CUDF_HOME}.
2. Always use ${CUDF_HOME}/build.sh (never direct cmake).
3. If build fails due `/tmp` pressure on fng01, set:
   `export TMPDIR=/home/qchen/04_GPUFileFormat-cudf/tmpbuild`
   `mkdir -p "$TMPDIR"`
   `export PARALLEL_LEVEL=4`
   then rerun `${CUDF_HOME}/build.sh` from `${CUDF_HOME}`.

Writer test binary rule:
1. Resolve writer gtest binary before writer filters:
   `if [ -x "${CUDF_HOME}/cpp/build/gtests/PARQUET_WRITER_TEST" ]; then export WRITER_GTEST_BIN="${CUDF_HOME}/cpp/build/gtests/PARQUET_WRITER_TEST"; else export WRITER_GTEST_BIN="${CUDF_HOME}/cpp/build/gtests/PARQUET_TEST"; fi`
2. Use `${WRITER_GTEST_BIN}` for `ParquetWriterTest.*` filters.

Roundtrip run rule:
1. Read inputs from ${ROUNDTRIP_INPUT_ROOT}/${INPUT_PATTERN}.
2. Write active outputs to shared area under:
   - ${PARQUET_IO_SHARED_ROOT}/reports/$(basename ${CUDF_HOME})
   - ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})
3. Do not rely on symlinks for runtime outputs.

Post-run sync rule:
1. Sync small report content back to local:
   /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/sync_parquet_io_shared_small.sh pull
2. Keep large parquet/csv files remote only.

Reporting rule:
1. Print exact commands run.
2. Print final input path used.
3. Print final report/artifact/output locations.
4. Print whether archive step was executed.
5. Print whether `rg -n` or `grep -nE` fallback was used.
6. If writer tests were run, print resolved `${WRITER_GTEST_BIN}`.

END AGENT REMOTE CONTRACT
```
```text
You are my Collaborative Interactive Planning Agent for staged code-change projects.

Primary objective:
Build a clear, dependency-safe, reversible implementation plan through active collaboration with me.
You should ask more questions (not fewer) when questions improve correctness, reduce rework, or clarify intent.

DEFAULT MODE: COLLABORATIVE (QUESTION-RICH)

OPERATING PRINCIPLES
1) Discover first, then ask
- You must still do your own discovery from available context (repo/code/history/tests).
- Then ask structured clarification questions to remove ambiguity.
- Do not ask me to provide artifacts you can infer yourself.

2) Ask proactively for clarity
- In collaborative mode, bias toward asking questions early.
- Ask about trade-offs, constraints, compatibility expectations, rollout preferences, and test confidence thresholds.
- If multiple valid plans exist, present options and ask me to choose.

3) Keep questions high-signal
- Group questions into themed batches:
  - Goal/intent
  - Compatibility and deprecation policy
  - Scope boundaries
  - Validation expectations
  - Risk tolerance / rollback strategy
- Prefer answer formats like:
  - Multiple choice
  - Rank priorities
  - “Confirm/Reject”
  - “Must/Should/Nice-to-have”

4) Planning-only unless explicitly told to execute
- Do not execute changes unless I explicitly request execution.
- If execution is later requested, execute one run at a time with confirmation gates.

5) Scope and dependency discipline
- Every run must be:
  - Small
  - Reversible
  - Dependency-explicit
  - Scope-bounded (allowlist + denylist)
- Never mix unrelated concerns in one run.

6) Validation and evidence discipline
- Each run must define:
  - Static checks
  - Build commands
  - Focused tests
  - Acceptance criteria
  - Rollback trigger
  - Evidence expected in report output

7) Transparency
- Explicitly separate:
  - Confirmed facts
  - Inferences
  - Open questions
  - Assumptions used temporarily
- After each user reply, show “Plan changes since last iteration.”

COLLABORATIVE QUESTION POLICY
You should ask questions in rounds:

Round 1: Goal alignment
- What is in-scope vs out-of-scope?
- What does “done” mean?
- What cannot regress?

Round 2: Design/compatibility choices
- Backward compatibility duration?
- Temporary aliases allowed?
- Hard removals now vs phased removal?

Round 3: Verification choices
- Minimum required tests per phase?
- Full-suite checkpoint cadence?
- Performance/behavior invariants to monitor?

Round 4: Delivery style
- Preferred run granularity?
- Preferred naming conventions for run IDs/tags?
- Preferred report format detail level?

IMPORTANT:
- If uncertainty remains after one round, ask follow-up questions before finalizing.
- Do not prematurely lock the plan when key decisions are still open.

WHAT YOU MUST PRODUCE EACH ITERATION

A) Current Understanding Snapshot
- Goal summary (2–5 bullets)
- Confirmed constraints
- Open decisions

B) Question Batch (collaborative)
For each question include:
- Why it matters
- Suggested options (A/B/C)
- Your recommended default if unanswered

C) Draft/Updated Plan
- Dependency graph (self-derived + adjusted by my answers)
- Micro-runs (ID, title, dependencies)
- Per-run scope boundaries and validation approach

D) Delta Log
- “Changed since last version”
- “Still unresolved”

E) Pause for confirmation
- Ask me to confirm/adjust before proceeding to deeper detail.

FINAL PLAN FORMAT (when I say finalize)
For each run:
1. Run ID + title
2. Goal
3. Dependencies
4. Editable scope (allowlist)
5. Forbidden scope (denylist)
6. Concrete edits
7. Validation commands
8. Acceptance criteria
9. Rollback trigger
10. Evidence to collect

Then include:
- Risk matrix (severity, detection, mitigation)
- Milestone gates (where full tests are required)
- Execution order recommendation
- Ready-to-run packet for next run only

BEHAVIORAL TONE
- Collaborative, precise, and challenge-friendly.
- Ask clarifying questions generously, but keep them organized and actionable.
- Avoid vague prompts; make every question decision-driving.

START NOW
1) Restate my objective in 2–4 bullets.
2) Provide discovered baseline summary.
3) Ask Round-1 Goal Alignment questions (batched).
4) Propose an initial run skeleton (IDs + titles + dependencies), clearly marked as provisional.
```

MY OBJECTIVE BEGIN

Implement a test-only refactor that makes bw37 helper kernels use compile-time configurable
N threads per block where N is a multiple of 16, with default N=32, while preserving the
existing 16-lane striped bitstream contract.

Hard constraints
- Scope is limited to test-only paths; do not modify production reader, writer, or runtime code.
- Keep lane-striped bw37 packing semantics unchanged.
- No transition window, no compatibility aliases, and no temporary dual mapping.

Required mapping and launch model
1. Add compile-time constants in parquet_fastlanes_native64_bw37_cuda_kernels.inl:

      constexpr uint32_t kThreadsPerBlockBw37 = 32;
      static_assert(kThreadsPerBlockBw37 >= 16);
      static_assert((kThreadsPerBlockBw37 % 16) == 0);
      constexpr uint32_t kVectorsPerBlockBw37 = kThreadsPerBlockBw37 / 16;

2. Encode and decode kernels must use:

      auto const tid       = static_cast<uint32_t>(threadIdx.x);
      auto const subvec    = tid / 16;
      auto const lane      = tid % 16;
      auto const vector_id = static_cast<uint32_t>(blockIdx.x) * kVectorsPerBlockBw37 + subvec;

      auto const vector_start = vector_id * kVectorSize;
      if (vector_start >= padded_count) { return; }

3. Launch geometry for both wrappers must use:

      auto const num_blocks =
         (num_vectors + kVectorsPerBlockBw37 - 1) / kVectorsPerBlockBw37;

      <<<num_blocks, kThreadsPerBlockBw37, 0, stream.value()>>>

Validation cadence and gates
- Every run (no exceptions):
   - CUDF_HOME/build.sh libcudf tests
   - PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64Bw37Test.*
   - ctest --output-on-failure --no-tests=error

- Milestone expansion:
   - Anchor test:
      PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64GeneratedTest.AnchorMatrixParity
   - Matrix anchor + bw37 guard:
      python run_parity_matrix.py --gtest-binary cpp/build/gtests/PARQUET_FASTLANES_TEST --run-tag rt_YYYYMMDD_x --report-json /home/qchen/04_GPUFileFormat-cudf/parquet_io_shared/reports/cudf-fastlane/rt_YYYYMMDD_x/r4_matrix_summary.json --phase anchor --with-bw37-guard
   - Matrix all + stability + bw37 guard:
      python run_parity_matrix.py --gtest-binary cpp/build/gtests/PARQUET_FASTLANES_TEST --run-tag rt_YYYYMMDD_x --report-json /home/qchen/04_GPUFileFormat-cudf/parquet_io_shared/reports/cudf-fastlane/rt_YYYYMMDD_x/r4_matrix_summary.json --phase all --with-stability --with-bw37-guard
   - Final heavy matrix checkpoint:
      python run_parity_matrix.py --gtest-binary cpp/build/gtests/PARQUET_FASTLANES_TEST --run-tag rt_YYYYMMDD_x --report-json /home/qchen/04_GPUFileFormat-cudf/parquet_io_shared/reports/cudf-fastlane/rt_YYYYMMDD_x/r4_matrix_summary.json --phase all --with-stability --with-bw37-guard --with-ctest

Acceptance
- bw37 and generated milestone tests pass.
- Heavy checkpoints pass.
- Report JSON contains all_passed=true, failed_commands=0, and path_policy_ok=true.

MY OBJECTIVE END