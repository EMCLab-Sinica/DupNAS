# TFLM Template

This project builds TensorFlow Lite Micro libraries and a simple runner for your models.

## Setup
Install dependencies before building:
```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Build
Run `make` to generate static libraries for both the host (Linux) and Cortex-M7 (with CMSIS-NN):
```
make
```
The build pulls in TFLM sources and produces the libraries under `gen/`, including:
- `gen/linux_x86_64_debug_gcc/lib/libtensorflow-microlite.a`
- `gen/cortex_m_generic_cortex-m7+fp_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a`

To add your own models, drop `.tflite` files into `src/models`. The generator will include them on the next `make` and expose per-model entrypoints named `tflm_main_<model_name>`, each taking `(uint8_t* tensor_arena, int tensor_arena_size, uint32_t (*get_time_ms)())` so you can supply your own arena buffer and an millisecond time getter (e.g., `HAL_GetTick` on STM32).

## Run models on host
After `make`, execute the host runner to invoke all configured models:
```
./tflm_main
```

You can override the default 300KB tensor arena used by the host runner by passing `TENSOR_ARENA_SIZE=<bytes>` to `make`.
