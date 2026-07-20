#!/usr/bin/env bash
set -euo pipefail

# Run all ONNX models in this model folder for one fixed VM setting.
# Expected layout:
#   sample_onnx/
#     DupNAS_SA.py
#     shufflenet/run_allsamples_vmXX.sh
#     mobilenet/run_allsamples_vmXX.sh
#     inception/run_allsamples_vmXX.sh
#
# Usage:
#   bash run_allsamples_vmXX.sh [output_root] [python_cmd]
# Example:
#   bash run_allsamples_vmXX.sh /path/to/output_dir python3

VM_SIZE=96
TIME_LIMIT="${TIME_LIMIT:-180}"

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"       # sample_onnx/<model>
MODEL_NAME="$(basename "$SCRIPT_DIR")"                    # shufflenet/mobilenet/inception
SAMPLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"               # sample_onnx
ONNX_DIR="$SCRIPT_DIR"
PY_SCRIPT="$SAMPLE_ROOT/DupNAS_SA.py"

# OUTPUT_ROOT="${1:-${OUTPUT_ROOT:-$SAMPLE_ROOT/output_dir}}"
PY_RAW="${2:-${PYTHON:-}}"
OUTDIR="$SAMPLE_ROOT/outputs/$MODEL_NAME/vm$VM_SIZE"
mkdir -p "$OUTDIR"

pick_python() {
  if [[ -n "$PY_RAW" ]]; then
    echo "$PY_RAW"
    return
  fi
  for c in python3.9 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return
    fi
  done
  if command -v py >/dev/null 2>&1; then
    echo "py -3.9"
    return
  fi
  echo ""
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  echo "[ERROR] No python found in PATH."
  exit 1
fi
read -r -a PY_CMD <<< "$PY"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] DupNAS_SA.py not found at: $PY_SCRIPT"
  exit 1
fi

shopt -s nullglob
onnx_files=("$ONNX_DIR"/*.onnx)
shopt -u nullglob

if (( ${#onnx_files[@]} == 0 )); then
  echo "[ERROR] No .onnx found in: $ONNX_DIR"
  exit 1
fi

selected_list="$OUTDIR/selected_vm${VM_SIZE}.txt"
: > "$selected_list"
for f in "${onnx_files[@]}"; do
  base="$(basename "$f" .onnx)"
  marker="$OUTDIR/${base}_pdq_config_detail_VM${VM_SIZE}_goal_bal.txt"
  if [[ -f "$marker" ]]; then
    continue
  fi
  echo "$f" >> "$selected_list"
done
mapfile -t selected_onnx < <(sort -V "$selected_list")

TAG="vm${VM_SIZE}"
CSV="$OUTDIR/${MODEL_NAME}_${TAG}_dupnasa_run_times.csv"
SUMMARY="$OUTDIR/${MODEL_NAME}_${TAG}_dupnasa_summary_times.csv"

echo "model,vm,dupnasa_ms,status" > "$CSV"

dupnas_sum=0
dupnas_n=0
fail_n=0

now_ms() {
  if date +%s%N >/dev/null 2>&1; then
    echo $(( $(date +%s%N) / 1000000 ))
  else
    "${PY_CMD[@]}" - <<'PY'
import time
print(int(time.time() * 1000))
PY
  fi
}

safe_avg() {
  local sum="$1"
  local n="$2"
  if (( n > 0 )); then
    echo $(( sum / n ))
  else
    echo -1
  fi
}

echo "[Info] MODEL_NAME = $MODEL_NAME"
echo "[Info] VM_SIZE    = $VM_SIZE KB"
echo "[Info] ONNX_DIR   = $ONNX_DIR"
echo "[Info] PY_SCRIPT  = $PY_SCRIPT"
echo "[Info] OUTDIR     = $OUTDIR"
echo "[Info] Found ${#onnx_files[@]} ONNX files"
echo "[Info] Selected ${#selected_onnx[@]} files after skipping existing DupNAS outputs"

if (( ${#selected_onnx[@]} == 0 )); then
  echo "[Info] Nothing to do."
  exit 0
fi

pushd "$OUTDIR" >/dev/null
for f in "${selected_onnx[@]}"; do
  base="$(basename "$f" .onnx)"
  marker="$OUTDIR/${base}_pdq_config_detail_VM${VM_SIZE}_goal_bal.txt"
  if [[ -f "$marker" ]]; then
    echo "==> $base (VM=$VM_SIZE) [SKIP]"
    continue
  fi

  echo "==> $base (VM=$VM_SIZE)"
  start="$(now_ms)"
  if timeout "${TIME_LIMIT}s" "${PY_CMD[@]}" "$PY_SCRIPT" \
      --onnx "$f" \
      --mode dupnas \
      --priority bal \
      --vmsize "$VM_SIZE" \
      --output_dir "$OUTDIR" \
      --export_file; then
    elapsed=$(( $(now_ms) - start ))
  else
    elapsed=-1
  fi

  if (( elapsed >= 0 )); then
    dupnas_sum=$((dupnas_sum + elapsed))
    dupnas_n=$((dupnas_n + 1))
    echo "    dupnas: ${elapsed} ms"
    echo "$base,$VM_SIZE,$elapsed,ok" >> "$CSV"
  else
    fail_n=$((fail_n + 1))
    echo "    [TIMEOUT/FAIL] dupnas"
    echo "$base,$VM_SIZE,-1,fail" >> "$CSV"
  fi

  # tinyTS and patchTS are intentionally disabled for this DupNAS-only run.
  # To re-enable tinyTS: run --mode tinyts --priority mem.
  # To re-enable patchTS: run --mode patchts --priority bal.
done
popd >/dev/null

dupnas_avg="$(safe_avg "$dupnas_sum" "$dupnas_n")"
{
  echo "metric,dupnasa_ms,dupnasa_n,fail_n,dupnasa_avg_ms"
  echo "totals,$dupnas_sum,$dupnas_n,$fail_n,$dupnas_avg"
} > "$SUMMARY"

echo "================== DONE =================="
echo "Outputs saved under: $OUTDIR"
echo "CSV     : $CSV"
echo "SUMMARY : $SUMMARY"
echo "=========================================="
