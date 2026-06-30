set -euo pipefail

case "$1" in
    tflm_f7)
        bash utils/run_tflm_f7.sh configs/fig11/tflite.txt
        ;;
    tflm_h7)
        bash utils/run_tflm_h7.sh configs/fig11/tflite.txt
        ;;
    cubeai_f7)
        bash utils/run_cubeai_f7.sh configs/fig11/onnx.txt
        ;;
    cubeai_h7)
        bash utils/run_cubeai_h7.sh configs/fig11/onnx.txt
        ;;
    *)
        echo "unknown target: $1"
        exit 1
        ;;
esac
