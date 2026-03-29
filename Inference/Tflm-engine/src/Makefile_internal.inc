MODEL_DIR := $(EXTERNAL_DIR)models/
MODEL_TFLITE := $(wildcard $(MODEL_DIR)*.tflite)
GEN_DIR := $(EXTERNAL_DIR)gen/
GEN_SCRIPT := $(EXTERNAL_DIR)gen_models.py

MICROLITE_CC_SRCS += $(GEN_DIR)models.cc $(EXTERNAL_DIR)tflm_main.cc

.PHONY: clean_gen

$(GEN_DIR)models.cc $(GEN_DIR)models.h: $(GEN_SCRIPT) $(MODEL_DIR) $(MODEL_TFLITE)
	python $(GEN_SCRIPT) --models_dir $(MODEL_DIR) --out_dir $(GEN_DIR)

$(CORE_OBJDIR)$(GEN_DIR)models.o: $(GEN_DIR)models.cc $(GEN_DIR)models.h
$(CORE_OBJDIR)$(EXTERNAL_DIR)tflm_main.o: $(EXTERNAL_DIR)tflm_main.cc $(EXTERNAL_DIR)tflm_main.h $(GEN_DIR)models.h

clean_gen:
	rm -rf $(GEN_DIR)

clean: clean_gen
