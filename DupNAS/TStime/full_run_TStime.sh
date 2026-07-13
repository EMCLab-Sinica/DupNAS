
#!/usr/bin/env bash
set -euo pipefail

# Randomly sample ONNX models from sample_onnx/<model>, then run DupNAS_SA.py
# for VM96, VM128, and VM256 sequentially.
#
# Sample sizes:
#   shufflenet: 2400
#   mobilenet : 3200
#   inception : 3200
#
# Usage:
#   cd TStime
#   bash random_run_TStime_large.sh <model> [python_cmd] [random_seed]
#
# Examples:
#   bash random_run_TStime_large.sh shufflenet python3.9 0
#   bash random_run_TStime_large.sh mobilenet python3.9 0
#   bash random_run_TStime_large.sh inception python3.9 0

TIME_LIMIT="${TIME_LIMIT:-180}"
VMS=(96 128 256)

MODEL_NAME="${1:-}"
PY_RAW="${2:-${PYTHON:-}}"
RANDOM_SEED="${3:-0}"

if [[ -z "$MODEL_NAME" ]]; then
  echo "[ERROR] Missing model argument."
  echo "Usage: bash random_run_TStime_large.sh <shufflenet|mobilenet|inception> [python_cmd] [random_seed]"
  exit 1
fi

# Set model-specific sample number
case "$MODEL_NAME" in
  shufflenet)
    SAMPLE_N=2400
    ;;
  mobilenet)
    SAMPLE_N=3200
    ;;
  inception)
    SAMPLE_N=3200
    ;;
  *)
    echo "[ERROR] Unknown model: $MODEL_NAME"
    echo "Allowed: shufflenet, mobilenet, inception"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLE_ROOT="$PROJECT_ROOT/sample_onnx"
ONNX_DIR="$SAMPLE_ROOT/$MODEL_NAME"
PY_SCRIPT="$SCRIPT_DIR/DupNAS_SA.py"
BASE_OUTDIR="$SCRIPT_DIR/$MODEL_NAME"

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

if [[ ! -d "$ONNX_DIR" ]]; then
  echo "[ERROR] ONNX directory not found: $ONNX_DIR"
  exit 1
fi

shopt -s nullglob
onnx_files=("$ONNX_DIR"/*.onnx)
shopt -u nullglob

if (( ${#onnx_files[@]} == 0 )); then
  echo "[ERROR] No .onnx found in: $ONNX_DIR"
  exit 1
fi

mkdir -p "$BASE_OUTDIR"

SELECTED_LIST="$BASE_OUTDIR/${MODEL_NAME}_sample${SAMPLE_N}_seed${RANDOM_SEED}_selected.txt"

# Randomly select SAMPLE_N models using Python for deterministic sampling.
"${PY_CMD[@]}" - "$ONNX_DIR" "$SAMPLE_N" "$RANDOM_SEED" "$SELECTED_LIST" <<'PY'
import random
import sys
from pathlib import Path

onnx_dir = Path(sys.argv[1])
sample_n = int(sys.argv[2])
seed = int(sys.argv[3])
out_file = Path(sys.argv[4])

files = sorted(onnx_dir.glob("*.onnx"), key=lambda p: p.name)

if not files:
    raise SystemExit(f"[ERROR] No .onnx files found in {onnx_dir}")

rng = random.Random(seed)

if len(files) <= sample_n:
    selected = files
else:
    selected = rng.sample(files, sample_n)
    selected = sorted(selected, key=lambda p: p.name)

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(
    "\n".join(str(p) for p in selected) + "\n",
    encoding="utf-8"
)

print(f"[Info] Random selected {len(selected)} / {len(files)} ONNX files")
print(f"[Info] Selected list: {out_file}")
PY

mapfile -t selected_onnx < "$SELECTED_LIST"

if (( ${#selected_onnx[@]} == 0 )); then
  echo "[ERROR] Selected ONNX list is empty."
  exit 1
fi

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

ALL_TIMES_FILE="$BASE_OUTDIR/${MODEL_NAME}_random${SAMPLE_N}_seed${RANDOM_SEED}_all_vm_runtime_ms.txt"
: > "$ALL_TIMES_FILE"

TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_SUM=0
TOTAL_MIN=""
TOTAL_MAX=""

update_runtime_stats() {
  local elapsed="$1"

  if (( elapsed >= 0 )); then
    TOTAL_OK=$((TOTAL_OK + 1))
    TOTAL_SUM=$((TOTAL_SUM + elapsed))

    if [[ -z "$TOTAL_MIN" || "$elapsed" -lt "$TOTAL_MIN" ]]; then
      TOTAL_MIN="$elapsed"
    fi

    if [[ -z "$TOTAL_MAX" || "$elapsed" -gt "$TOTAL_MAX" ]]; then
      TOTAL_MAX="$elapsed"
    fi

    echo "$elapsed" >> "$ALL_TIMES_FILE"
  else
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi
}

echo "[Info] MODEL_NAME  = $MODEL_NAME"
echo "[Info] SCRIPT_DIR  = $SCRIPT_DIR"
echo "[Info] SAMPLE_ROOT = $SAMPLE_ROOT"
echo "[Info] ONNX_DIR    = $ONNX_DIR"
echo "[Info] PY_SCRIPT   = $PY_SCRIPT"
echo "[Info] PY          = $PY"
echo "[Info] TIME_LIMIT  = ${TIME_LIMIT}s"
echo "[Info] SAMPLE_N    = $SAMPLE_N"
echo "[Info] RANDOM_SEED = $RANDOM_SEED"
echo "[Info] Output base = $BASE_OUTDIR"

echo "================== RUN START =================="

for VM_SIZE in "${VMS[@]}"; do
  OUTDIR="$BASE_OUTDIR/vm$VM_SIZE"
  mkdir -p "$OUTDIR"

  CSV="$OUTDIR/${MODEL_NAME}_vm${VM_SIZE}_dupnas_run_times.csv"
  SUMMARY="$OUTDIR/${MODEL_NAME}_vm${VM_SIZE}_dupnas_summary_times.csv"

  echo "model,vm,dupnas_ms,status" > "$CSV"

  vm_sum=0
  vm_ok=0
  vm_fail=0
  vm_min=""
  vm_max=""

  echo "---------------- VM${VM_SIZE} ----------------"
  echo "[Info] OUTDIR = $OUTDIR"
  echo "[Info] CSV    = $CSV"

  pushd "$OUTDIR" >/dev/null

  for f in "${selected_onnx[@]}"; do
    base="$(basename "$f" .onnx)"

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

      echo "    dupnas: ${elapsed} ms"
      echo "$base,$VM_SIZE,$elapsed,ok" >> "$CSV"

      vm_sum=$((vm_sum + elapsed))
      vm_ok=$((vm_ok + 1))

      if [[ -z "$vm_min" || "$elapsed" -lt "$vm_min" ]]; then
        vm_min="$elapsed"
      fi

      if [[ -z "$vm_max" || "$elapsed" -gt "$vm_max" ]]; then
        vm_max="$elapsed"
      fi

      update_runtime_stats "$elapsed"

    else
      elapsed=-1

      echo "    [TIMEOUT/FAIL] dupnas"
      echo "$base,$VM_SIZE,-1,fail" >> "$CSV"

      vm_fail=$((vm_fail + 1))
      update_runtime_stats "$elapsed"
    fi
  done

  popd >/dev/null

  if (( vm_ok > 0 )); then
    vm_avg=$((vm_sum / vm_ok))
  else
    vm_avg=-1
    vm_min=-1
    vm_max=-1
  fi

  {
    echo "metric,value"
    echo "model,$MODEL_NAME"
    echo "vm,vm$VM_SIZE"
    echo "selected_models,${#selected_onnx[@]}"
    echo "ok_count,$vm_ok"
    echo "fail_count,$vm_fail"
    echo "min_ms,$vm_min"
    echo "avg_ms,$vm_avg"
    echo "max_ms,$vm_max"
  } > "$SUMMARY"

  echo "[VM Summary] $MODEL_NAME vm$VM_SIZE"
  echo "  ok_count : $vm_ok"
  echo "  fail_cnt : $vm_fail"
  echo "  min_ms   : $vm_min"
  echo "  avg_ms   : $vm_avg"
  echo "  max_ms   : $vm_max"
  echo "  CSV      : $CSV"
  echo "  SUMMARY  : $SUMMARY"
done

if (( TOTAL_OK > 0 )); then
  TOTAL_AVG=$((TOTAL_SUM / TOTAL_OK))
else
  TOTAL_AVG=-1
  TOTAL_MIN=-1
  TOTAL_MAX=-1
fi

TOTAL_SAMPLES=$(( ${#selected_onnx[@]} * ${#VMS[@]} ))

OVERALL_SUMMARY="$SCRIPT_DIR/${MODEL_NAME}_total${TOTAL_SAMPLES}runs_summary.csv"

{
  echo "metric,value"
  echo "model,$MODEL_NAME"
  echo "sample_n,${#selected_onnx[@]}"
  echo "vm_settings,96|128|256"
  echo "expected_results,$TOTAL_SAMPLES"
  echo "ok_count,$TOTAL_OK"
  echo "fail_count,$TOTAL_FAIL"
  echo "min_ms,$TOTAL_MIN"
  echo "avg_ms,$TOTAL_AVG"
  echo "max_ms,$TOTAL_MAX"
} > "$OVERALL_SUMMARY"

fmt_sec() {
  local ms="$1"

  if (( ms < 0 )); then
    echo "N/A"
  else
    awk -v ms="$ms" 'BEGIN { printf "%.3f", ms / 1000.0 }'
  fi
}

TOTAL_MIN_SEC="$(fmt_sec "$TOTAL_MIN")"
TOTAL_AVG_SEC="$(fmt_sec "$TOTAL_AVG")"
TOTAL_MAX_SEC="$(fmt_sec "$TOTAL_MAX")"

echo "================== ALL VM DONE =================="
echo "[Overall TS Runtime among $TOTAL_SAMPLES results]"
echo "  model      : $MODEL_NAME"
echo "  min        : ${TOTAL_MIN_SEC} sec (${TOTAL_MIN} ms)"
echo "  avg        : ${TOTAL_AVG_SEC} sec (${TOTAL_AVG} ms)"
echo "  max        : ${TOTAL_MAX_SEC} sec (${TOTAL_MAX} ms)"
echo "  selected   : $SELECTED_LIST"
echo "  summary    : $OVERALL_SUMMARY"
echo "================================================="

