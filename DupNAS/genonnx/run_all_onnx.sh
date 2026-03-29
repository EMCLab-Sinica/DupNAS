#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# run_all_onnx.sh
# Run all ONNX files in the current folder
# - mode is decided by model name:
#     dupnas  -> --mode dupnas
#     tinyts  -> --mode tinyts
#     patchts -> --mode patchts
#     nots    -> --mode nots
# - vmsize is decided by model name:
#     vm96    -> --vmsize 96
#     vm128   -> --vmsize 128
#     vm256   -> --vmsize 256
# - create inferred_onnx/ if it does not exist
# ==================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
OUTDIR="$SCRIPT_DIR"
ONNX_DIR="$SCRIPT_DIR"
PY_SCRIPT="$SCRIPT_DIR/DupNAS_SA.py"

TIME_LIMIT=300   # seconds

# --------------------------------------------------
# Python selection, can be overridden by $1
# --------------------------------------------------
PY_RAW="${1:-}"

pick_python() {
  if [[ -n "$PY_RAW" ]]; then echo "$PY_RAW"; return; fi
  if [[ -n "${PYTHON:-}" ]]; then echo "${PYTHON}"; return; fi
  for c in python3.9 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return; fi
  done
  if command -v py >/dev/null 2>&1; then echo "py -3.9"; return; fi
  echo ""
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  echo "[ERROR] No python found in PATH."
  exit 1
fi

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------
echo "[Info] OUTDIR   = $OUTDIR"
echo "[Info] ONNX_DIR = $ONNX_DIR"
echo "[Info] PY       = $PY"
echo "[Info] PY_SCRIPT= $PY_SCRIPT"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] baseline_tinyts.py not found at: $PY_SCRIPT"
  exit 1
fi

# --------------------------------------------------
# Make inferred_onnx dir if not exist
# --------------------------------------------------
mkdir -p "$ONNX_DIR/inferred_onnx"

# --------------------------------------------------
# Time helper (ms)
# --------------------------------------------------
now_ms() {
  if date +%s%N >/dev/null 2>&1; then
    echo $(( $(date +%s%N) / 1000000 ))
  else
    $PY - <<'PY'
import time
print(int(time.time()*1000))
PY
  fi
}

# --------------------------------------------------
# Collect all ONNX in current folder only
# --------------------------------------------------
mapfile -t onnx_files < <(
  find "$ONNX_DIR" -maxdepth 1 -type f -iname "*.onnx" | sort -V
)

if (( ${#onnx_files[@]} == 0 )); then
  echo "[ERROR] No ONNX files found in current folder: $ONNX_DIR"
  exit 1
fi

echo "[Info] Found ${#onnx_files[@]} ONNX files in current folder"

# --------------------------------------------------
# CSV output
# --------------------------------------------------
CSV="$OUTDIR/run_times.csv"
echo "model,mode,vmsize,time_ms,status" > "$CSV"

# --------------------------------------------------
# Helpers to infer mode / vm from model name
# --------------------------------------------------
infer_mode() {
  local name="$1"
  local lname
  lname="$(echo "$name" | tr '[:upper:]' '[:lower:]')"

  if [[ "$lname" == *"dupnas"* ]]; then
    echo "dupnas"
  elif [[ "$lname" == *"tinyts"* ]]; then
    echo "tinyts"
  elif [[ "$lname" == *"patchts"* ]]; then
    echo "patchts"
  elif [[ "$lname" == *"nots"* ]]; then
    echo "nots"
  else
    echo ""
  fi
}

infer_vmsize() {
  local name="$1"
  local lname
  lname="$(echo "$name" | tr '[:upper:]' '[:lower:]')"

  if [[ "$lname" == *"vm96"* ]]; then
    echo "96"
  elif [[ "$lname" == *"vm128"* ]]; then
    echo "128"
  elif [[ "$lname" == *"vm256"* ]]; then
    echo "256"
  else
    echo ""
  fi
}

# --------------------------------------------------
# Run inside OUTDIR so --export_file outputs go here
# --------------------------------------------------
pushd "$OUTDIR" >/dev/null

for f in "${onnx_files[@]}"; do
  base="$(basename "$f" .onnx)"
  mode="$(infer_mode "$base")"
  vmsize="$(infer_vmsize "$base")"

  echo "==> $base"

  if [[ -z "$mode" ]]; then
    echo "    [SKIP] Cannot infer mode from model name"
    echo "$base,,-1,-1,skip_no_mode" >> "$CSV"
    continue
  fi

  if [[ -z "$vmsize" ]]; then
    echo "    [SKIP] Cannot infer vmsize from model name"
    echo "$base,$mode,-1,-1,skip_no_vmsize" >> "$CSV"
    continue
  fi

  echo "    mode   = $mode"
  echo "    vmsize = $vmsize"

  start=$(now_ms)
  if timeout ${TIME_LIMIT}s $PY "$PY_SCRIPT" \
      --onnx "$base" \
      --mode "$mode" \
      --priority bal \
      --vmsize "$vmsize" \
      --export_file; then
    elapsed=$(( $(now_ms) - start ))
    echo "    [OK] ${elapsed} ms"
    echo "$base,$mode,$vmsize,$elapsed,ok" >> "$CSV"
  else
    echo "    [TIMEOUT/FAIL]"
    echo "$base,$mode,$vmsize,-1,fail" >> "$CSV"
  fi
done

popd >/dev/null

echo
echo "================== DONE =================="
echo "Outputs (export_file) saved under: $OUTDIR"
echo "CSV     : $CSV"
echo "=========================================="
echo -e "\a"

echo
echo "Press ENTER to exit..."
read -r