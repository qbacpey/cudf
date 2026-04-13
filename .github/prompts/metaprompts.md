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

Here is a comprehensive, structured prompt designed specifically to be fed into a planning LLM. It gives the LLM all the context, constraints, and explicit step-by-step tasks it needs to correctly refactor both the encoding and decoding paths.

***

**Copy and paste the following prompt to your Planning LLM:**

```text
# SYSTEM ARCHITECTURE & REFACTORING PLAN: Parquet FastLanes Encoders & Decoders

## Context & The Problem
We need to refactor the cuDF Parquet FastLanes integration to eliminate "Template Fatigue." Currently, the encoder implementation (`fastlanes_encode.cu/.cuh`) uses heavy templating (`encode_page_impl<T>`) to mash together completely different execution models. The decoder is similarly overloaded. 

We are introducing a new GPU-native 64-bit encoding, which means we now have three fundamentally distinct execution paths that must be explicitly separated in both the encoding and decoding stages:

1. **32-bit CPU Path:** Uses `Encoding::FASTLANE_BITPACK_RAW`. Data is encoded on the CPU. The kernel does not apply a delta internally.
2. **64-bit Split32 CPU Path:** Uses `Encoding::FASTLANE_BITPACK_SPLIT64`. Data is encoded on the CPU by splitting 64-bit integers into two 32-bit streams.
3. **64-bit Native GPU Path (NEW):** Uses `Encoding::FASTLANES_DELTA_BINARY`. Data is encoded entirely on the GPU. The encoding/decoding kernels handle the delta internally.

## Your Objective
Please create a detailed implementation plan and write the code to cleanly separate these three paths. 

### Intent 1: Refactor the Public Encoder API (`fastlanes_encode.cuh`)
- **Intent:** Make the API explicitly reflect the three execution paths. 
- **Action:** Keep `FastLanesInt32Encoder`. Rename the existing 64-bit encoder to `FastLanesInt64Split32Encoder`. Create a new `FastLanesInt64NativeEncoder`. All three should share the same public method signatures (`encode_page` and `encode_pages`).

### Intent 2: De-template the Encoder Implementation (`fastlanes_encode.cu`)
- **Intent:** Remove the generic `encode_page_impl<T>` and `encode_pages_impl<T>`. Stop using `if constexpr` to switch between CPU and GPU workflows.
- **Action:** Create three dedicated, non-templated helper functions in the anonymous namespace for single-page and batch encoding. Map the three classes from Intent 1 directly to these specific helpers. 

### Intent 3: Implement the Native64 GPU Encoder
- **Intent:** The new `FASTLANES_DELTA_BINARY` path must be 100% GPU-accelerated. No input data should be downloaded to the host.
- **Action:** Design the Native64 helper to use the `native64_generated` API. It must compute the minimum value (base bits) on the GPU, compute the maximum delta and bitwidth on the GPU (e.g., using Thrust/CUB), allocate the final device buffer (including space for the 128-byte header), pack the data on the GPU, and finally generate and copy the header from the CPU to the device buffer.

### Intent 4: Separate the Decode Kernels (`page_decode.cuh` / `fastlanes.cu`)
- **Intent:** Currently, the decode dispatcher launches an INT32 and an INT64 kernel, which try to dynamically figure out what to do. We want a strict 1-to-1 mapping between the three encoding types and their respective CUDA kernels.
- **Action:** 
  1. Refactor the page validation logic (`setup_and_validate_fastlanes_page`) to properly identify and route `FASTLANE_BITPACK_RAW`, `FASTLANE_BITPACK_SPLIT64`, and `FASTLANES_DELTA_BINARY` based on the page metadata.
  2. Create three explicitly named `__global__` decode kernels, one for each encoding type.
  3. Implement the new Native64 decode kernel using the `native64_generated` decoding API. Remember that this specific format handles the delta internally, unlike the raw bitpack.
  4. Update the host launch function (`decode_fastlanes_binary`) to dispatch to all three kernels, relying on the internal kernel validation to allow threads to early-exit if the page doesn't match their designated encoding.

Please analyze this architecture, confirm your understanding of the separation of concerns, and then provide the refactored code for both the encoder and decoder.
MY OBJECTIVE END