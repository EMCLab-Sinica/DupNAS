#!/usr/bin/env bash
set -euo pipefail

OPTION="${OPTION:-}"

if [[ ! "$OPTION" =~ ^(shufflenet|mobilenet|inception)-(samples|full)$ ]]; then
  echo "Error: Invalid OPTION: $OPTION"
  echo "Allowed options:"
  echo "  shufflenet-samples"
  echo "  shufflenet-full"
  echo "  mobilenet-samples"
  echo "  mobilenet-full"
  echo "  inception-samples"
  echo "  inception-full"
  exit 1
fi

MODEL="${BASH_REMATCH[1]}"
RUN_TYPE="${BASH_REMATCH[2]}"

echo "Running model: $MODEL"
echo "Run type: $RUN_TYPE"

cd DupNAS/TStime

case "$RUN_TYPE" in
  samples)
    bash random_run_TStime.sh "$MODEL" python3.9 0
    ;;
  full)
    bash full_run_TStime.sh "$MODEL" python3.9 0
    ;;
esac

