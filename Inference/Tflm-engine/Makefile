JOBS ?= 8
TENSOR_ARENA_SIZE ?= 307200

TFLM_COMMIT := 6c1c1a8
TFLM_MAKEFILE := tflite-micro/tensorflow/lite/micro/tools/make/Makefile

.PHONY: all microlite microlite-m7 dummy clean

all: microlite microlite-m7 tflm_main

tflite-micro:
	test -d tflite-micro || git clone https://github.com/tensorflow/tflite-micro.git
	git -C tflite-micro checkout $(TFLM_COMMIT)

microlite: tflite-micro
	set -e; \
	restore() { git -C tflite-micro restore tensorflow/lite/micro/tools/make/Makefile; }; \
	trap restore EXIT; \
	sed -E -i 's/-O[1-3sz]/-O0/g' "$(TFLM_MAKEFILE)"; \
	$(MAKE) -j$(JOBS) -f "$(TFLM_MAKEFILE)" TENSORFLOW_ROOT=tflite-micro/ EXTERNAL_DIR=src/ BUILD_TYPE=debug microlite

microlite-m7: tflite-micro
	$(MAKE) -j$(JOBS) -f "$(TFLM_MAKEFILE)" TENSORFLOW_ROOT=tflite-micro/ EXTERNAL_DIR=src/ OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_generic TARGET_ARCH=cortex-m7+fp FPU=fpv4-sp-d16 microlite

tflm_main: microlite main.cpp
	g++ main.cpp \
	  -Wall \
	  -DTENSOR_ARENA_SIZE=$(TENSOR_ARENA_SIZE) \
	  -Itflite-micro \
	  -Lgen/linux_x86_64_debug_gcc/lib \
	  -ltensorflow-microlite \
	  -o tflm_main

dummy: microlite main.cpp
	g++ main.cpp \
	  -Wall \
	  -DTENSOR_ARENA_SIZE=$(TENSOR_ARENA_SIZE) \
	  -DTF_LITE_STATIC_MEMORY \
	  -Itflite-micro \
	  -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	  -Itflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	  -Lgen/linux_x86_64_debug_gcc/lib \
	  -ltensorflow-microlite \
	  -o /dev/null

clean:
	$(MAKE) -f "$(TFLM_MAKEFILE)" TENSORFLOW_ROOT=tflite-micro/ EXTERNAL_DIR=src/ clean
	rm -rf tflm_main
