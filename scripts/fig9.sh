set -euo pipefail

[[ "$OPTION" =~ ^(shufflenet|mobilenet|inception)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

MODEL="$OPTION"

echo "Executing Model: $MODEL..."

cd DupNAS/TStime
bash random_run_TStime.sh "$MODEL" python3.9 0
