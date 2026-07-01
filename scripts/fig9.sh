set -euo pipefail

[[ "$OPTION" =~ ^(shufflenet|mobilenet|inception)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

MODEL="${BASH_REMATCH[1]}"

echo "Running model: $MODEL..."

cd DupNAS/TStime
bash random_run_TStime.sh "$MODEL" python3.9 0
