#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

cp settings/settings-shuffle.py settings.py
python3.9 -m NASBase.spec_onnx_gen

cd "$ROOT_DIR/genonnx"
cp DupNAS_SA.py gen_ts_cfg.py run_all_onnx.sh run_ts_convert.sh shuffle/
sed -i 's/\r$//' shuffle/run_all_onnx.sh shuffle/run_ts_convert.sh

cd "$ROOT_DIR/genonnx/shuffle"
bash run_all_onnx.sh
