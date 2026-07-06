set -euo pipefail

cd tflm-template
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
make tflm_main
./tflm_main 2> >(tee host.txt)
