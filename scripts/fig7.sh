set -euo pipefail

[[ "$OPTION" =~ ^(shuffle|mbv2|incept)$ ]] || { echo "Error: Invalid OPTION"; exit 1; }

MODEL="${BASH_REMATCH[1]}"

echo "Running model: $MODEL..."

cd DupNAS
python3.9 verify_by_image100_acc.py \
  --onnx-dirs genonnx/val_onnx/${MODEL} \
  --output genonnx/onnx_image100_accuracy_${MODEL}_test.csv \
  --batch-size 1 \
  --provider cpu \
