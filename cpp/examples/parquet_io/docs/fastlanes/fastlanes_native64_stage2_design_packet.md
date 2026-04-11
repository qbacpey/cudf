# FL64-R9 Stage 2 Design Packet (Design-Only)

## Status
- Stage: Stage 2 planning only
- Code implementation: intentionally out of scope for this packet
- Baseline dependency: FL64-R8 validation evidence from Stage 1 test-only work

## Scope
This packet defines the production integration plan for native64 FastLanes decode with rollback controls.
No production source files are modified in this round.

## Inputs and Anchors
- Header/runtime contract anchor: cpp/include/cudf/fastlanes/common.cuh
- Native64 decode integration anchor: cpp/src/io/parquet/page_fastlanes_decode.cu
- Stage 1 matrix/report anchor: cpp/examples/parquet_io/tools/tests/run_parity_matrix.py
- Stage 1 test harnesses:
  - cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
  - cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu

## Goal
Add production-safe native64 runtime support for INT64 FastLanes pages while preserving split32 behavior and existing diagnostics.

## Non-Goals
- No removal of split32 decode path in initial production landing.
- No change to writer defaults in first production patch.
- No schema/path changes in Stage 1 matrix reporting.

## Current State Summary
- common.cuh currently validates external mode using a split64_mode boolean and checks body size against expected_body_size_bytes(split64_mode).
- page_fastlanes_decode.cu currently launches split32 INT64 decode and carries TODO markers for layout-based split32 vs native64 dispatch.
- Stage 1 confirms test-only parity across bw0..64 with bw37 guards.

## Proposed Production Architecture

### 1. Header Contract Extension (common.cuh)
Introduce explicit layout classification for INT64 payloads without breaking existing serialized pages.

Design direction:
- Keep existing PageHeader byte layout stable for backward compatibility in first release.
- Add a layout classifier helper derived from existing header fields to discriminate:
  - RAW scalar32
  - SPLIT64 split32 streams
  - NATIVE64 single-stream payload
- Extend validation helpers so mode checks are layout-aware instead of boolean-only.

Planned API shape (design target, not yet implemented):
- enum class external_layout_mode : uint8_t { raw32, split64, native64, invalid };
- classify_external_layout(PageHeader const&) -> external_layout_mode
- expected_body_size_bytes_for_layout(PageHeader const&, external_layout_mode)
- is_valid_for_external_layout(PageHeader const&, external_layout_mode)

Compatibility note:
- Existing SPLIT64 pages must remain valid without re-encoding.

### 2. Decode Dispatch Split (page_fastlanes_decode.cu)
Replace single INT64 split32 launch with layout-based dispatch after page header validation.

Design direction:
- Keep current split32 decode kernel intact as baseline path.
- Add native64 INT64 decode kernel path that consumes one packed stream per vector.
- Host launch wrapper dispatches by classified layout mode per page.

Dispatch behavior (target):
- If layout == split64: current split32 kernel path.
- If layout == native64: native64 kernel path.
- Else: fail fast with existing decode error plumbing.

Safety constraints:
- Preserve one warp per block and one block per page in first integration.
- Preserve existing null/nesting restrictions in setup_and_validate_fastlanes_page.
- Preserve error surface and stream semantics.

### 3. Writer and Runtime Rollout Policy
Two-step rollout to limit blast radius:
- Step A: reader-only support for native64 pages (opt-in decode support).
- Step B: optional writer emission policy after dedicated compatibility soak.

## Risk Matrix
| Risk | Severity | Detection | Mitigation |
|---|---|---|---|
| Header misclassification causes wrong kernel path | High | Targeted native64/split64 mixed-page tests | Explicit classifier tests and fail-fast invalid branch |
| Body-size validation mismatch for native64 pages | High | Header unit tests and decode preflight checks | Dedicated expected size helper per layout |
| Split32 regression during dispatch migration | High | Existing split32 regression suite plus ctest checkpoint | Keep split32 kernel unchanged and branch-isolated |
| Stream behavior regression | Medium | stream-identification ctest and parity tests | Keep stream-bound launch and sync pattern unchanged |
| Diagnostics regression (bw37 observability) | Medium | bw37 guard suite in release gates | Keep bw37 tests mandatory in migration gates |

## Migration Gates (Design)
1. Gate P1: Header classifier and validation helpers land with unit tests.
2. Gate P2: Decode dispatch switch lands with split32 baseline preserved.
3. Gate P3: Native64 decode parity gates pass alongside split32 regressions.
4. Gate P4: Full ctest checkpoint plus targeted parquet IO regression pass.
5. Gate P5: Optional writer-path proposal reviewed separately.

## Rollback Conditions
Immediate rollback of production native64 path if any condition occurs:
- Split32 decode regression appears in existing parquet tests.
- Header classifier returns ambiguous layout for valid historical pages.
- ctest checkpoint fails in stream-identification mode for modified components.

## Verification Plan (Post-Implementation)
- Targeted gtests:
  - native64 decode path tests (new)
  - split32 backward-compat tests (existing + new mixed cases)
  - bw37 guard suite
- Matrix runner:
  - keep Stage 1 matrix schema/path contract unchanged
  - require all_passed=true and path_policy_ok=true
- Full checkpoint:
  - ctest --output-on-failure --no-tests=error

## Open Questions
- Should native64 layout be represented via explicit flag bits or inferred from bitwidth/component structure in first release?
- Is per-page mixed layout support required immediately, or can first rollout assume chunk-homogeneous layout?
- What is the writer default policy once native64 decode is stable: remain split32 by default or feature-flag native64 output?

## Deliverables in This Round
- This design packet only.
- No source implementation changes.
