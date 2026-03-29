#pragma once

#include "gen/models.h"

#ifdef __cplusplus
extern "C" {
#endif

void register_debug_log_callback(void (*callback)(const char* s));
void set_print_output(int enable);

#define DECLARE_TFLM_MAIN(symbol, display_name) \
  int tflm_main_##symbol(uint8_t* tensor_arena, int tensor_arena_size, uint32_t (*get_time_ms)());
TFLM_FOREACH_MODEL(DECLARE_TFLM_MAIN)
#undef DECLARE_TFLM_MAIN

int tflm_main(uint8_t* tensor_arena, int tensor_arena_size, uint32_t (*get_time_ms)());

#ifdef __cplusplus
}
#endif
