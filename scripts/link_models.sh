set -euo pipefail

for model in shufflenet mobilenet inception; do
    ln -s /4TB/aeuser/DupNAS-AE/DupNAS/sample_onnx/$model/*.onnx DupNAS/sample_onnx/$model/
done
