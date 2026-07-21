set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 MODELS_TXT flash,ram,latency,accuracy"
    exit 1
fi

export PATH="/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin:$PATH"
PROGRAMMER="/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin/STM32_Programmer_CLI"

PROJECT_DIR="stm_projects/tflm_f7"
HEADER_FILE="$PROJECT_DIR/Core/Inc/run_tflm.h"
BUILD_DIR="$PROJECT_DIR/STM32CubeIDE/Release"
ELF_FILE="$BUILD_DIR/tflm.elf"

SERIAL_NUMBER="066EFF535570514867224209"
TTY_DEVICE="/dev/ttyACM0"

MODELS_TXT="$1"
METRICS="$2"
TFLM_MODELS_DIR="tflm-template/src/models"
RESULTS_DIR="$PWD/results/tflm_f7"
RESULTS_CSV="$RESULTS_DIR/results.csv"
ACCURACY_CSV="$(dirname "$0")/accuracy.csv"

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
echo "model_name,$METRICS" > "$RESULTS_CSV"

scale_metric() {
    [ "$1" = "NA" ] && echo "NA" || awk -v value="$1" 'BEGIN {printf "%.2f\n", value / 1000}'
}

metric_unit() {
    case "$1" in
        flash|ram) echo "KB" ;;
        latency) echo "s" ;;
    esac
}

run_model() {
    local MODEL_NAME="$1"
    local SANITIZED_NAME="$2"
    local RUN_DIR="$3"

    (
        set -e
        echo "tflm_main_${SANITIZED_NAME}(tensor_arena, TENSOR_ARENA_SIZE, HAL_GetTick);" > "$HEADER_FILE"

        echo "==== BUILD ===="
        make -C "$BUILD_DIR" clean
        make -C "$BUILD_DIR" -j8 all

        echo "=== FLASH ==="
        "$PROGRAMMER" -c port=SWD sn="$SERIAL_NUMBER" -w "$ELF_FILE" -v -rst
        sleep 10s
    )

    if [ $? -eq 0 ]; then
        echo "=== SUCCESS ==="
    else
        echo "=== FAIL ==="
    fi
}

while IFS= read -r MODEL || [ -n "$MODEL" ]; do
    [ -z "$MODEL" ] && continue

    if [ -f "$TFLM_MODELS_DIR/$MODEL" ]; then
        MODEL_FILE="$TFLM_MODELS_DIR/$MODEL"
    else
        echo "Model not found: $MODEL"
        continue
    fi

    echo "Running $MODEL_FILE ..."

    MODEL_NAME=$(basename "${MODEL_FILE%.*}")
    RUN_DIR="$RESULTS_DIR/$MODEL_NAME"
    mkdir -p "$RUN_DIR"

    SANITIZED_NAME="${MODEL_NAME//[^a-zA-Z0-9_]/_}"

    BOARD_LOG="$RUN_DIR/board.log"
    HOST_LOG="$RUN_DIR/host.log"

    stty -F "$TTY_DEVICE" 115200 raw
    cat "$TTY_DEVICE" > "$BOARD_LOG" &
    RECORD_PID=$!

    run_model "$MODEL_NAME" "$SANITIZED_NAME" "$RUN_DIR" > "$HOST_LOG" 2>&1

    kill "$RECORD_PID"

    FLASH=$(grep "${MODEL_NAME}.*started" tflm-template/host.txt 2>/dev/null | awk '{print $3}' | tr -d "(" || true)
    FLASH=${FLASH:-NA}
    FLASH=$(scale_metric "$FLASH")

    RAM=$(grep -A 1 "${MODEL_NAME}.*started" tflm-template/host.txt 2>/dev/null | grep "Arena allocation total" | awk '{print $5}' || true)
    RAM=${RAM:-NA}
    RAM=$(scale_metric "$RAM")

    LATENCY=$(grep "completed" "$BOARD_LOG" | awk '{print $4}' || true)
    LATENCY=${LATENCY:-NA}
    LATENCY=$(scale_metric "$LATENCY")

    ACCURACY_NAME="${MODEL_NAME%_full_integer_quant}"
    ACCURACY_NAME="${ACCURACY_NAME%_quantized}"
    ACCURACY=$(awk -F, -v model="$ACCURACY_NAME" '$1 == model {print $2; exit}' "$ACCURACY_CSV" || true)
    ACCURACY=${ACCURACY:-NA}

    ROW="$MODEL_NAME"
    SUMMARY=""
    for METRIC in ${METRICS//,/ }; do
        VAR="${METRIC^^}"
        VALUE="${!VAR}"
        UNIT=$(metric_unit "$METRIC")
        DISPLAY="${VALUE}${UNIT:+ $UNIT}"
        ROW="$ROW,$VALUE"
        SUMMARY="${SUMMARY:+$SUMMARY, }$METRIC: $DISPLAY"
    done

    echo "$SUMMARY"
    echo "$ROW" >> "$RESULTS_CSV"
done < "$MODELS_TXT"

echo "Completed. Results saved to $RESULTS_CSV"
