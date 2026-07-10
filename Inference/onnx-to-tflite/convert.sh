#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mbv2|shuffle|incept>"
    exit 1
fi

ARC="$1"
SOURCE_DIR="../Model-converter/ts_converted/${ARC}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[ERROR] Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

shopt -s nullglob
ONNX_FILES=("$SOURCE_DIR"/*.onnx)
shopt -u nullglob

if [ "${#ONNX_FILES[@]}" -eq 0 ]; then
    echo "[ERROR] No ONNX files found under: $SOURCE_DIR"
    exit 1
fi

echo "[INFO] Copying ${#ONNX_FILES[@]} ONNX file(s) from:"
echo "       $SOURCE_DIR"
echo "       -> $(pwd)"

cp -v "${ONNX_FILES[@]}" .

sudo docker run --rm \
    -v "$(pwd)":/workdir \
    -w /workdir \
    ghcr.io/pinto0309/onnx2tf:1.28.5 \
    bash -c 'find . -maxdepth 1 -name "*.onnx" -exec onnx2tf -i {} -oiqt \;'

mkdir -p outputs
mv saved_model/*_full_integer_quant.tflite outputs/
rm -rf saved_model

echo "[DONE] Converted TFLite files are under: $(pwd)/outputs"
