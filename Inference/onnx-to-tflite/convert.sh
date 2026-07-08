#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
sudo docker run --rm -v "$(pwd)":/workdir -w /workdir ghcr.io/pinto0309/onnx2tf:1.28.5 bash -c 'find . -name "*.onnx" -exec onnx2tf -i {} -oiqt \;'

mkdir -p outputs
mv saved_model/*_full_integer_quant.tflite outputs/
rm -rf saved_model
