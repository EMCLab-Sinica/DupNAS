
#!/usr/bin/env bash
set -euo pipefail

OPTION="${1:-${OPTION:-}}"

case "${OPTION}" in
  stage1+2|stage3+4|full-stage)
    ;;
  *)
    echo "Error: Invalid OPTION: '${OPTION}'"
    echo "Valid options: stage1+2, stage3+4, full-stage"
    exit 1
    ;;
esac

echo "Selected option: ${OPTION}"

# =========================
# User configuration
# =========================
KAGGLE_API_TOKEN="<your-token>"

ARC="shuffle"
MODE="dupnas"
VMSIZE="128"
SUFFIX="aetest"

# =========================
# Persistent prerequisite files
# =========================
PERSIST_TRAIN_LOG="/4TB/aeuser/DupNAS-AE/DupNAS/NASBase/train_log"

# Paths relative to DupNAS/, after "cd DupNAS/"
LOCAL_TRAIN_LOG="NASBase/train_log"
LOCAL_CKPT_LOG="NASBase/checkpoints"

# =========================
# Configure Kaggle access
# =========================
export KAGGLE_API_TOKEN

mkdir -p ~/.kaggle
echo "${KAGGLE_API_TOKEN}" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# =========================
# Prepare DupNAS
# =========================
cd DupNAS/

cp "settings/settings-${ARC}.py" settings.py

LOG_PREFIX="${ARC}-im100-${MODE}-vm${VMSIZE}-${SUFFIX}"

# =========================
# Prerequisite restore helpers
# =========================
copy_required_file() {
  local src="$1"
  local dst_dir="$2"

  if [[ ! -f "$src" ]]; then
    echo "ERROR: Required prerequisite file not found:"
    echo "  $src"
    exit 1
  fi

  mkdir -p "$dst_dir"
  cp "$src" "$dst_dir/"

  echo "Copied prerequisite:"
  echo "  $src"
  echo "  -> $dst_dir/"
}

restore_supernet_files() {
  local result_name="${SUFFIX}_trsupnetresults.json"
  local ckpt_name="${SUFFIX}_supernet_${ARC}_best.pth"

  local persist_result="${PERSIST_TRAIN_LOG}/${result_name}"
  local persist_ckpt="${PERSIST_TRAIN_LOG}/${ckpt_name}"

  local local_result="${LOCAL_TRAIN_LOG}/${result_name}"
  local local_ckpt="${LOCAL_CKPT_LOG}/${ckpt_name}"

  # Copy Stage 2 result JSON
  copy_required_file \
    "${persist_result}" \
    "${LOCAL_TRAIN_LOG}"

  # Copy Stage 2 supernet checkpoint
  copy_required_file \
    "${persist_ckpt}" \
    "${LOCAL_CKPT_LOG}"

  # Get absolute path in the current GitHub Actions workspace
  local local_ckpt_abs
  local_ckpt_abs="$(realpath "${local_ckpt}")"

  # Rewrite the checkpoint path stored in the copied JSON
  python3.9 - "${local_result}" "${local_ckpt_abs}" <<'PY'
import json
import sys

json_path = sys.argv[1]
checkpoint_path = sys.argv[2]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

old_path = data.get("supernet_best_ckpt")
data["supernet_best_ckpt"] = checkpoint_path

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Updated supernet checkpoint path:")
print(f"  old: {old_path}")
print(f"  new: {checkpoint_path}")
PY

  echo "Restored supernet files successfully:"
  echo "  JSON: ${local_result}"
  echo "  CKPT: ${local_ckpt_abs}"
}

restore_stage3_prerequisites() {
  echo "=============================="
  echo "Restoring Stage 1/2 prerequisites"
  echo "=============================="

  # Stage 1 output
  copy_required_file \
    "${PERSIST_TRAIN_LOG}/${SUFFIX}_ssoptlog.json" \
    "${LOCAL_TRAIN_LOG}"

  # Stage 2 result JSON + supernet checkpoint
  restore_supernet_files

  echo "Prerequisite restoration completed."
}

# =========================
# Stage functions
# =========================
run_stage1() {
  echo "=============================="
  echo "Starting Stage 1"
  echo "=============================="

  python3.9 -m NASBase.run_nas \
    --stages 1 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    2>&1 | tee "${LOG_PREFIX}-s1.txt"

  echo "Stage 1 finished successfully."
}

run_stage2() {
  echo "=============================="
  echo "Starting Stage 2"
  echo "=============================="

  python3.9 -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29601 \
    -m NASBase.run_nas \
    --stages 2 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    --dist ddp \
    --amp fp16 \
    2>&1 | tee "${LOG_PREFIX}-s2.txt"

  echo "Stage 2 finished successfully."
}

run_stage3() {
  echo "=============================="
  echo "Starting Stage 3"
  echo "=============================="

  python3.9 -m NASBase.run_nas \
    --stages 3 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    2>&1 | tee "${LOG_PREFIX}-s3.txt"

  echo "Stage 3 finished successfully."
}

run_stage4() {
  echo "=============================="
  echo "Starting Stage 4"
  echo "=============================="

  python3.9 -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29611 \
    -m NASBase.run_nas \
    --stages 4 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    --dist ddp \
    --amp fp16 \
    2>&1 | tee "${LOG_PREFIX}-s4.txt"

  echo "Stage 4 finished successfully."
}

# =========================
# Run selected option
# =========================
case "${OPTION}" in
  stage1+2)
    # Stage 2 directly uses the Stage 1 output generated
    # in the same GitHub Actions workspace.
    run_stage1
    run_stage2

    echo "=============================="
    echo "Stages 1 and 2 finished successfully."
    echo "=============================="
    ;;

  stage3+4)
    # Restore Stage 1/2 outputs from the persistent server copy.
    # Stage 4 then directly uses the Stage 3 output generated
    # in this same GitHub Actions workspace.
    restore_stage3_prerequisites
    run_stage3
    run_stage4

    echo "=============================="
    echo "Stages 3 and 4 finished successfully."
    echo "=============================="
    ;;

  full-stage)
    # All stages run sequentially in the same workspace.
    # No prerequisite restoration is needed.
    run_stage1
    run_stage2
    run_stage3
    run_stage4

    echo "=============================="
    echo "All stages finished successfully."
    echo "=============================="
    ;;
esac
