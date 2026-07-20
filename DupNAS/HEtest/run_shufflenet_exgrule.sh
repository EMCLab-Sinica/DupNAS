#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# Auto run DupNAS multiseed random combo count
# Model: shufflenet
#
# Usage:
#   ./run_shufflenet_exgrule.sh <EXGRULE> <VM_SETTING>
#
# Examples:
#   ./run_shufflenet_exgrule.sh neither 96
#   ./run_shufflenet_exgrule.sh PConly 128
#   ./run_shufflenet_exgrule.sh BPonly 256
#
# New EXGRULE mapping:
#   neither = old both
#   PConly  = old boundary
#   BPonly  = old branches
# ==================================================

MODEL="shufflenet"
EXGRULE="${1:-}"
VM_SETTING="${2:-}"

SEEDS="0,1,2,3,4,5,6,7,8,9"
SAMPLES=1000
SCRIPT="DupNAS_SA_exgrule.py"

if [[ -z "${EXGRULE}" || -z "${VM_SETTING}" ]]; then
    echo "[ERROR] Missing arguments."
    echo "Usage: $0 <EXGRULE: neither|PConly|BPonly> <VM_SETTING>"
    exit 1
fi

# Accept old names only as convenience, then normalize to the new names.
case "${EXGRULE}" in
    neither|PConly|BPonly) ;;
    both) EXGRULE="neither" ;;
    boundary) EXGRULE="PConly" ;;
    branches) EXGRULE="BPonly" ;;
    *)
        echo "[ERROR] Unsupported EXGRULE: ${EXGRULE}"
        echo "Allowed: neither, PConly, BPonly"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HETEST_DIR="${SCRIPT_DIR}"  # HEtest
PROJECT_DIR="$(cd "${HETEST_DIR}/.." && pwd)"
ROOT_DIR="${HETEST_DIR}"
ONNX_DIR="${PROJECT_DIR}/sample_onnx/${MODEL}"
SUMMARY_CSV="${HETEST_DIR}/${MODEL}_vm${VM_SETTING}_peak_mem_summary.csv"
OUTDIR="${HETEST_DIR}/outputs/${EXGRULE}/${MODEL}/vm${VM_SETTING}"
RUN_LIST="$(mktemp)"

cleanup() {
    rm -f "${RUN_LIST}"
}
trap cleanup EXIT

if [[ ! -f "${ROOT_DIR}/${SCRIPT}" ]]; then
    echo "[ERROR] Cannot find script: ${ROOT_DIR}/${SCRIPT}"
    exit 1
fi

if [[ ! -f "${SUMMARY_CSV}" ]]; then
    echo "[ERROR] Cannot find summary CSV: ${SUMMARY_CSV}"
    echo "Expected all SUMMARY_CSV files under HEtest/."
    exit 1
fi

if [[ ! -d "${ONNX_DIR}" ]]; then
    echo "[ERROR] Cannot find ONNX directory: ${ONNX_DIR}"
    exit 1
fi

mkdir -p "${OUTDIR}"
mkdir -p "${ONNX_DIR}/inferred_onnx"

# Build run list from HEtest/<model>_vm<VM>_peak_mem_summary.csv.
# Keep the same selection logic as the previous BAT: valid_ourTS_bal == 0.
set +e
python3.9 - "${SUMMARY_CSV}" "${RUN_LIST}" <<'PYCSV'
import csv
import sys

summary_csv, run_list = sys.argv[1], sys.argv[2]
rows = []
with open(summary_csv, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    required = {'onnx_name', 'valid_ourTS_bal'}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in summary CSV: {sorted(missing)}")
    for row in reader:
        if str(row.get('valid_ourTS_bal', '')).strip() == '0':
            name = str(row.get('onnx_name', '')).strip()
            if name:
                rows.append(name)

if not rows:
    raise SystemExit(2)

with open(run_list, 'w', encoding='utf-8') as f:
    for name in rows:
        f.write(name + '\n')
PYCSV
status=$?
set -e
if [[ ${status} -eq 2 ]]; then
    echo "[INFO] No rows found with valid_ourTS_bal == 0 in ${SUMMARY_CSV}."
    exit 0
elif [[ ${status} -ne 0 ]]; then
    echo "[ERROR] Failed to build run list from ${SUMMARY_CSV}."
    exit 1
fi

COUNT=0
SKIP_COUNT=0
EXG_SUFFIX="_${EXGRULE}"

cd "${ROOT_DIR}"

echo
echo "=================================================="
echo "Model: ${MODEL}"
echo "EXG rule mode: ${EXGRULE}"
echo "VM setting: ${VM_SETTING} KB"
echo "Random seeds: ${SEEDS}"
echo "Random samples per seed: ${SAMPLES}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Project dir: ${PROJECT_DIR}"
echo "ONNX dir: ${ONNX_DIR}"
echo "Output dir: ${OUTDIR}"
echo "Run condition: valid_ourTS_bal == 0"
echo "=================================================="

while IFS= read -r ONNX_NAME || [[ -n "${ONNX_NAME}" ]]; do
    [[ -z "${ONNX_NAME}" ]] && continue

    REL_ONNX_BASE="../sample_onnx/${MODEL}/${ONNX_NAME}"
    ONNX_FILE="${ONNX_DIR}/${ONNX_NAME}.onnx"

    COUNT_OUT="${OUTDIR}/${ONNX_NAME}_combo_count_VM${VM_SETTING}${EXG_SUFFIX}.txt"
    ALLSEEDS_OUT="${OUTDIR}/${ONNX_NAME}_combo_random_samples_ALLSEEDS_VM${VM_SETTING}${EXG_SUFFIX}.txt"

    if [[ ! -f "${ONNX_FILE}" ]]; then
        echo "[SKIP] ONNX file not found: ${ONNX_FILE}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    if [[ -f "${ALLSEEDS_OUT}" ]]; then
        echo "[SKIP] Already finished: ${ONNX_NAME}"
        echo "       Existing report: ${ALLSEEDS_OUT}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    COUNT=$((COUNT + 1))
    echo
    echo "=================================================="
    echo "[${COUNT}] Running: ${ONNX_FILE}"
    echo "ONNX name: ${ONNX_NAME}"
    echo "VM size: ${VM_SETTING} KB"
    echo "EXG rule mode: ${EXGRULE}"
    echo "Output dir: ${OUTDIR}"
    echo "=================================================="

    # Remove stale local reports and this model's inferred ONNX before running.
    rm -f "${ONNX_DIR}/${ONNX_NAME}_combo_count_VM${VM_SETTING}${EXG_SUFFIX}.txt"
    rm -f "${ONNX_DIR}/${ONNX_NAME}_combo_random_samples_ALLSEEDS_VM${VM_SETTING}${EXG_SUFFIX}.txt"
    rm -f "${ONNX_DIR}/inferred_onnx/${ONNX_NAME}_inferred.onnx"

    if ! python3.9 "${SCRIPT}" \
        --mode count \
        --onnx "${REL_ONNX_BASE}" \
        --vmsize "${VM_SETTING}" \
        --random_seeds "${SEEDS}" \
        --random_samples "${SAMPLES}" \
        --exgrule "${EXGRULE}"; then
        echo "[ERROR] Failed on ${ONNX_NAME}. Stop running."
        rm -f "${ONNX_DIR}/inferred_onnx/${ONNX_NAME}_inferred.onnx"
        exit 1
    fi

    # Collect compact reports into HEtest/outputs/<EXGRULE>/<model>/vm<VM_SETTING>.
    if [[ -f "${ONNX_DIR}/${ONNX_NAME}_combo_count_VM${VM_SETTING}${EXG_SUFFIX}.txt" ]]; then
        mv -f "${ONNX_DIR}/${ONNX_NAME}_combo_count_VM${VM_SETTING}${EXG_SUFFIX}.txt" "${COUNT_OUT}"
    else
        echo "[WARN] Missing count report for ${ONNX_NAME}"
    fi

    if [[ -f "${ONNX_DIR}/${ONNX_NAME}_combo_random_samples_ALLSEEDS_VM${VM_SETTING}${EXG_SUFFIX}.txt" ]]; then
        mv -f "${ONNX_DIR}/${ONNX_NAME}_combo_random_samples_ALLSEEDS_VM${VM_SETTING}${EXG_SUFFIX}.txt" "${ALLSEEDS_OUT}"
        echo "[COLLECT] ${ALLSEEDS_OUT}"
    else
        echo "[WARN] Missing ALLSEEDS report for ${ONNX_NAME}"
    fi

    # Do not keep all inferred ONNX files. They can be regenerated and are not needed after reports are collected.
    rm -f "${ONNX_DIR}/inferred_onnx/${ONNX_NAME}_inferred.onnx"

done < "${RUN_LIST}"

echo
echo "=================================================="
echo "Finished."
echo "Model: ${MODEL}"
echo "EXG rule mode: ${EXGRULE}"
echo "VM setting: ${VM_SETTING} KB"
echo "Total executed: ${COUNT}"
echo "Total skipped: ${SKIP_COUNT}"
echo "Reports saved to: ${OUTDIR}"
echo "=================================================="
