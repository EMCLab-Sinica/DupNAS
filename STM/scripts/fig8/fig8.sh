set -euo pipefail

[[ "$OPTION" =~ ^(shufflenet|mobilenet|inception)-vm(96|128)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

MODEL="${BASH_REMATCH[1]}"
VM="${BASH_REMATCH[2]}"

echo "Running model: $MODEL, VM: $VM..."

bash "scripts/run_tflm_f7.sh" "scripts/fig8/${MODEL}-vm${VM}.txt" latency,accuracy
