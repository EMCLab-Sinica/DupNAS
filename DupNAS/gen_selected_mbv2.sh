#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

cp settings/settings-mbv2.py settings.py
python3.9 -m NASBase.spec_onnx_gen

cd "$ROOT_DIR/genonnx"
cp DupNAS_SA.py gen_ts_cfg.py run_all_onnx.sh run_ts_convert.sh mbv2/
sed -i 's/\r$//' mbv2/run_all_onnx.sh mbv2/run_ts_convert.sh

cd "$ROOT_DIR/genonnx/mbv2"
bash run_all_onnx.sh
