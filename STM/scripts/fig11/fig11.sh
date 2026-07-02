set -euo pipefail

[[ "$OPTION" =~ ^(tflm|cubeai)-(f7|h7)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

ENGINE="${BASH_REMATCH[1]}"
BOARD="${BASH_REMATCH[2]}"

echo "Running engine: $ENGINE, board: $BOARD..."

bash "scripts/run_${ENGINE}_${BOARD}.sh" "scripts/fig11/${ENGINE}.txt" ram,latency

RESULTS_CSV="results/${ENGINE}_${BOARD}/results.csv"
python3 scripts/fig11/summarize_boxplot.py "$RESULTS_CSV"
