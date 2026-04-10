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

# FL64-R4: bw0..64 Generation Strategy Packet

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
| Native64 bw spec table (bw, words_per_lane, cross-boundary map) | Generated metadata (test-only) | FastLanes test owner | IO parquet owner | Script input spec | cpp/tests/io/fastlanes_native64_gen/spec/native64_bw_table.json | FL64-R4-M1 |
| CUDA kernel include (bw0..64, test-only) | Generated code | FastLanes test owner | CUDA reviewer | Generator + golden snapshots | cpp/tests/io/fastlanes_native64_gen/generated/native64_bw_kernels.inl | FL64-R4-M2 |
| CPU oracle compare harness glue | Handwritten test glue | FastLanes test owner | QA owner | Test source | cpp/tests/io/parquet_fastlanes_native64_generated_test.cu | FL64-R4-M2 |
| Parity matrix executor (anchor + full sweep) | Test utility | QA owner | FastLanes test owner | Matrix config | cpp/tests/io/fastlanes_native64_gen/tools/run_parity_matrix.py | FL64-R4-M3 |
| Run evidence summary | Report | QA owner | IO parquet owner | CI/remote logs | parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json | FL64-R4-M3 |
| Unresolved issues log | Design note | IO parquet owner | FastLanes test owner | This packet + run feedback | docs/cudf/source/developer_guide/fastlanes_native64_r4_generation_strategy.md | FL64-R4-M4 |

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


MY OBJECTIVE END