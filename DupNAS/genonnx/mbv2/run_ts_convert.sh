#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
MODEL_FAMILY="$(basename "$SCRIPT_DIR")"

# SCRIPT_DIR is expected to be:
#   DupNAS/genonnx/shuffle
#   DupNAS/genonnx/mbv2
#   DupNAS/genonnx/incept
DUPNAS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AE_ROOT="$(cd "$DUPNAS_ROOT/.." && pwd)"

CONVERTER_DIR="$AE_ROOT/Inference/Model-converter"
OUT_ROOT="$CONVERTER_DIR/ts_converted"
OUT_DIR="$OUT_ROOT/$MODEL_FAMILY"

PY="${PYTHON:-python3.9}"

echo "========== TS Convert =========="
echo "[INFO] Model family  : $MODEL_FAMILY"
echo "[INFO] Source dir    : $SCRIPT_DIR"
echo "[INFO] DupNAS root   : $DUPNAS_ROOT"
echo "[INFO] AE root       : $AE_ROOT"
echo "[INFO] Converter dir : $CONVERTER_DIR"
echo "[INFO] Output dir    : $OUT_DIR"
echo "[INFO] Python        : $PY"

mkdir -p "$OUT_DIR"
mkdir -p "$CONVERTER_DIR"

cd "$CONVERTER_DIR"

for onnx_path in "$SCRIPT_DIR"/*.onnx; do
    [ -e "$onnx_path" ] || {
        echo "[ERROR] No ONNX files found in $SCRIPT_DIR"
        exit 1
    }

    base="$(basename "$onnx_path" .onnx)"

    if [ -f "$SCRIPT_DIR/${base}.json" ]; then
        config_path="$SCRIPT_DIR/${base}.json"
    elif [ -f "$SCRIPT_DIR/${base}_config.json" ]; then
        config_path="$SCRIPT_DIR/${base}_config.json"
    elif [ -f "$SCRIPT_DIR/config.json" ]; then
        config_path="$SCRIPT_DIR/config.json"
    else
        echo "[WARN] Missing config for $base, skip"
        continue
    fi

    input_onnx="$CONVERTER_DIR/${base}.onnx"
    input_config="$CONVERTER_DIR/${base}_config.json"
    output_onnx="$CONVERTER_DIR/${base}_ts.onnx"

    echo "=================================================="
    echo "[RUN] $base"
    echo "      ONNX   : $onnx_path"
    echo "      Config : $config_path"

    cp "$onnx_path" "$input_onnx"
    cp "$config_path" "$input_config"

    "$PY" -m ts.cli "$input_onnx" "$input_config" "$output_onnx"

    cp "$output_onnx" "$OUT_DIR/${base}_ts.onnx"

    echo "[OK] Saved: $OUT_DIR/${base}_ts.onnx"
done

echo "=================================================="
echo "[DONE] TS conversion finished for $MODEL_FAMILY"
echo "[DONE] Converted models saved under: $OUT_DIR"