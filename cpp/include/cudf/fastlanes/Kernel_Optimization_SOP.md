Yes. For BW37 CUDA-argument experiments, the existing test stack is enough to validate correctness if you run it in a fixed order.

**Where Your BW37 Argument Change Must Propagate**
1. Kernel signatures live in parquet_fastlanes_native64_bw37_cuda_kernels.inl.
2. Kernel launch sites are in parquet_fastlanes_native64_bw37_cuda_kernels.inl.
3. Host wrappers that call those kernels are in parquet_fastlanes_native64_bw37_cuda_kernels.inl.
4. BW37 test callsites are in parquet_fastlanes_native64_bw37_test.cu.
5. The moved include is now consumed from parquet_fastlanes_native64_bw37_test.cu.

If you add/remove a kernel argument, update all 4 layers above or compilation/runtime will fail.

**Existing Tests You Should Use**
1. BW37 focused suite in parquet_fastlanes_native64_bw37_test.cu.
2. Generated native64 parity guard tests in parquet_fastlanes_native64_generated_test.cu, parquet_fastlanes_native64_generated_test.cu, parquet_fastlanes_native64_generated_test.cu.
3. Both are part of PARQUET_FASTLANES_TEST target in CMakeLists.txt.

**Full Post-Modification Workflow**
Run this after every BW37 change.

~~~bash
cat <<'EOF' | ssh qchen@fng01.lab.tuda.systems bash -s
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cudf_dev
export CPATH="$CONDA_PREFIX/include/rapids:$CONDA_PREFIX/include"

export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane
export PARQUET_IO_SHARED_ROOT=/home/qchen/04_GPUFileFormat-cudf/parquet_io_shared
export RUN_TAG=rt_20260410_f
export REPORT_JSON="$PARQUET_IO_SHARED_ROOT/reports/$(basename "$CUDF_HOME")/$RUN_TAG/r4_matrix_summary.json"

mkdir -p "$(dirname "$REPORT_JSON")"
cd "$CUDF_HOME"

# 1) Sanity: moved include is still referenced correctly
if command -v rg >/dev/null 2>&1; then
  rg -n "parquet_fastlanes_native64_bw37_cuda_kernels\.inl" \
    cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu \
    cpp/include/cudf/fastlanes/parquet_fastlanes_native64_bw37_cuda_kernels.inl
else
  grep -nE "parquet_fastlanes_native64_bw37_cuda_kernels\.inl" \
    cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu \
    cpp/include/cudf/fastlanes/parquet_fastlanes_native64_bw37_cuda_kernels.inl
fi

# 2) Build tests with /tmp fallback
build_with_fallback() {
  local target="$1"
  if ! "$CUDF_HOME/build.sh" $target; then
    export TMPDIR=/home/qchen/04_GPUFileFormat-cudf/tmpbuild
    mkdir -p "$TMPDIR"
    export PARALLEL_LEVEL=4
    "$CUDF_HOME/build.sh" $target
  fi
}

build_with_fallback "libcudf tests"

# 3) Fast correctness gates
cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64Bw37Test.*
cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64GeneratedTest.AnchorMatrixParity

# 4) Matrix runner gate (anchor + bw37 guard)
 /home/qchen/miniconda3/envs/cudf_dev/bin/python \
  cpp/examples/parquet_io/tools/tests/run_parity_matrix.py \
  --gtest-binary cpp/build/gtests/PARQUET_FASTLANES_TEST \
  --run-tag "$RUN_TAG" \
  --report-json "$REPORT_JSON" \
  --phase anchor \
  --with-bw37-guard

# 5) Print pass/fail summary
/home/qchen/miniconda3/envs/cudf_dev/bin/python - <<'PY'
import json
p = "/home/qchen/04_GPUFileFormat-cudf/parquet_io_shared/reports/cudf-fastlane/rt_20260410_f/r4_matrix_summary.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
print("all_passed=", d["summary"]["all_passed"])
print("failed_commands=", d["summary"]["failed_commands"])
print("path_policy_ok=", d["path_policy_ok"])
PY
EOF
~~~

**Deeper Validation (when fast gate passes)**
Use this to check that your change did not break non-BW37 native64 behavior.

~~~bash
cat <<'EOF' | ssh qchen@fng01.lab.tuda.systems bash -s
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cudf_dev
export CPATH="$CONDA_PREFIX/include/rapids:$CONDA_PREFIX/include"
export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane
export PARQUET_IO_SHARED_ROOT=/home/qchen/04_GPUFileFormat-cudf/parquet_io_shared
export RUN_TAG=rt_20260410_g
export REPORT_JSON="$PARQUET_IO_SHARED_ROOT/reports/$(basename "$CUDF_HOME")/$RUN_TAG/r4_matrix_summary.json"
mkdir -p "$(dirname "$REPORT_JSON")"
cd "$CUDF_HOME"

/home/qchen/miniconda3/envs/cudf_dev/bin/python \
  cpp/examples/parquet_io/tools/tests/run_parity_matrix.py \
  --gtest-binary cpp/build/gtests/PARQUET_FASTLANES_TEST \
  --run-tag "$RUN_TAG" \
  --report-json "$REPORT_JSON" \
  --phase all \
  --with-stability \
  --with-bw37-guard
EOF
~~~

If needed, append --with-ctest for the heaviest checkpoint, but that is much slower.

**How To Decide If Your Idea Is Correct**
1. Build succeeds with no signature/call mismatch.
2. ParquetFastLanesNative64Bw37Test.* passes.
3. AnchorMatrixParity passes.
4. Matrix JSON shows all_passed true, failed_commands 0, path_policy_ok true.
5. If you changed semantics (not just plumbing), run phase all with stability and compare baseline vs modified JSON summaries.

If you want, I can next give you a minimal A/B experiment template for one specific argument change (for example changing launch geometry or adding an extra kernel parameter) and the exact expected failure modes.