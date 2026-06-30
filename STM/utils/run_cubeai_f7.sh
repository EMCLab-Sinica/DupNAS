set -euo pipefail

if [ -z "$1" ]; then
    echo "usage: $0 MODELS_TXT"
    exit 1
fi

export PATH="/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin:$PATH"
STEDGEAI="/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
PROGRAMMER="/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin/STM32_Programmer_CLI"

PROJECT_DIR="stm_projects/cubeai_f7"
MY_MODEL_NAME_FILE="$PROJECT_DIR/Core/Inc/my_model_name.h"
MEMPOOL_FILE="$PROJECT_DIR/.ai/mempools.json"
BUILD_DIR="$PROJECT_DIR/STM32CubeIDE/Release"
ELF_FILE="$BUILD_DIR/manual.elf"

SERIAL_NUMBER="066EFF535570514867224209"
TTY_DEVICE="/dev/ttyACM0"

MODELS_TXT="$1"
ONNX_MODELS_DIR="onnx_models"
TFLM_MODELS_DIR="tflm-template/src/models"
RESULTS_DIR="$PWD/results/cubeai_f7"
RESULTS_CSV="$RESULTS_DIR/results.csv"

BATCH_SIZE="1"
OPT="balanced"
NAME="network"
VERBOSITY="1"
C_API="st-ai"
TARGET="stm32f7"

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
echo "model_name,flash,ram,latency" > "$RESULTS_CSV"

run_model() {
    local MODEL_FILE="$1"
    local MODEL_NAME="$2"
    local RUN_DIR="$3"
    local WS_DIR="$RUN_DIR/st_ai_ws"

    (
        set -e
        echo "static const char my_model_name[] = \"$MODEL_NAME\";" > "$MY_MODEL_NAME_FILE"

        echo "=== GENERATE ==="
        "$STEDGEAI" generate \
            --model "$MODEL_FILE" \
            --batch-size "$BATCH_SIZE" \
            --mode target \
            --optimization "$OPT" \
            --name "$NAME" \
            --verbosity "$VERBOSITY" \
            --c-api "$C_API" \
            --target "$TARGET" \
            --workspace "$WS_DIR" \
            --output "$PROJECT_DIR" \
            --memory-pool "$MEMPOOL_FILE" \
            --quiet
        rm -rf "$WS_DIR"

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

    if [ -f "$ONNX_MODELS_DIR/$MODEL" ]; then
        MODEL_FILE="$ONNX_MODELS_DIR/$MODEL"
    elif [ -f "$TFLM_MODELS_DIR/$MODEL" ]; then
        MODEL_FILE="$TFLM_MODELS_DIR/$MODEL"
    else
        echo "Model not found: $MODEL"
        continue
    fi

    echo "Running $MODEL_FILE ..."

    MODEL_NAME=$(basename "${MODEL_FILE%.*}")
    RUN_DIR="$RESULTS_DIR/$MODEL_NAME"
    mkdir -p "$RUN_DIR"

    BOARD_LOG="$RUN_DIR/board.log"
    HOST_LOG="$RUN_DIR/host.log"

    stty -F "$TTY_DEVICE" 115200 raw
    cat "$TTY_DEVICE" > "$BOARD_LOG" &
    RECORD_PID=$!

    run_model "$MODEL_FILE" "$MODEL_NAME" "$RUN_DIR" > "$HOST_LOG" 2>&1

    kill "$RECORD_PID"

    FLASH=$(grep "weights (ro)" "$HOST_LOG" | awk '{print $4}' | tr -d ",")
    FLASH=${FLASH:-NA}

    RAM=$(grep "activations (rw)" $HOST_LOG | awk '{print $4}' | tr -d ",")
    RAM=${RAM:-NA}

    LATENCY=$(grep "duration DWT" $BOARD_LOG | awk '{print $5}')
    LATENCY=${LATENCY:-NA}

    echo "flash: $FLASH, ram: $RAM, latency: $LATENCY"
    echo "$MODEL_NAME,$FLASH,$RAM,$LATENCY" >> "$RESULTS_CSV"
done < "$MODELS_TXT"

echo "All jobs finished."
