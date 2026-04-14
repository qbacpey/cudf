## Plan: FastLanes Native64 TODO Refactor

Third-iteration collaborative draft after Round-1, Round-2, and Round-3 alignment. Scope includes newly introduced TODO/FIXME items plus hidden debt in the same touched functions.

Confirmed constraints and choices:
- Keep full ctest green baseline (114/114).
- Keep PARQUET_FASTLANES_TEST native64/split64 coverage stable.
- Risk profile: balanced.
- Sentinel dtype mismatch and broad INT64/UINT64 activation are optional side tracks only.
- Decoder strategy: full dispatch redesign early.
- Writer cleanup granularity: two-step (in-file extraction, then file split).
- Debug policy: preserve runtime toggles and tighten to one debug API boundary.
- No temporary compatibility wrappers during refactor.
- Validation cadence: full ctest after every run.
- Acceptance strictness: functional equivalence plus stable encoded header/payload invariants in focused checks.
- Rollback triggers: any focused-test failure, any new decode/encode error path hit, any increased reserved-size overflow risk.
- Optional tracks O1/O2 deferred until all core runs complete.

Introduced TODO/FIXME items in touched FastLanes areas:
- cpp/src/fastlanes/fastlanes.cu:89
- cpp/src/fastlanes/fastlanes.cu:222
- cpp/src/fastlanes/fastlanes.cu:371
- cpp/src/fastlanes/fastlanes.cu:565
- cpp/src/io/parquet/page_enc.cu:3858
- cpp/src/io/parquet/page_enc.cu:4312
- cpp/src/io/parquet/page_fastlanes_decode.cu:40
- cpp/src/io/parquet/page_fastlanes_decode.cu:211
- cpp/src/io/parquet/page_fastlanes_decode.cu:506
- cpp/src/io/parquet/page_fastlanes_decode.cu:606
- cpp/src/io/parquet/reader_impl.cpp:304

Hidden adjacent debt included by scope:
- Native64 encode cast-mode asymmetry in cpp/src/fastlanes/fastlanes.cu.
- Debug/logging asymmetry and clustering in cpp/src/io/parquet/page_enc.cu and cpp/src/fastlanes/fastlanes.cu.
- Mixed host/device staging semantics in native64 encode path.

Execution order recommendation (provisional, now validation-complete policy aware):
1. R0 Baseline and guard freeze.
2. R1 Native64 decode correctness fix at page_fastlanes_decode.cu:506 plus assertions.
3. R2 Early decode dispatch redesign at reader_impl.cpp:304 and page_fastlanes_decode.cu:606.
4. R3 Native64 decode API boundary cleanup at page_fastlanes_decode.cu:40 and :211.
5. R4 Writer cleanup step 1 (in-file extraction) in page_enc.cu at :3858 and :4312.
6. R5 Writer cleanup step 2 (file split) after R4 stability.
7. R6 Encoder helper dedup in fastlanes.cu at :89, :222, :371.
8. R7 Native64 staging cleanup + cast-mode parity hardening in fastlanes.cu at :565 and nearby logic.
9. O1 Optional sentinel dtype mismatch packet (deferred).
10. O2 Optional broader native64 activation packet (deferred).

Per-run verification template (for execution-ready packets in next iteration):
- Build from CUDF_HOME with build.sh only.
- Run focused touched-path tests with explicit filters.
- Run full ctest checkpoint immediately after each run.
- Capture invariant checks for header/body mode metadata on touched paths.
- Roll back run immediately on any configured rollback trigger.

Open items for finalization step:
- Exact focused gtest filter sets per run (to be locked in run packets).
- Exact invariant assertions per run (header fields, body_size, mode/pre_delta expectations).