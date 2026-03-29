#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tflm_main.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {
bool g_print_outputs = false;

int CalculateElementCount(const TfLiteTensor* tensor) {
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i)
    count *= tensor->dims->data[i];
  return count;
}

void FillInput(TfLiteTensor* input) {
  const int total_size = CalculateElementCount(input);
  switch (input->type) {
    case kTfLiteFloat32:
      for (int i = 0; i < total_size; ++i)
        input->data.f[i] = static_cast<float>(i % 7);
      break;
    case kTfLiteInt8:
      for (int i = 0; i < total_size; ++i)
        input->data.int8[i] = static_cast<int8_t>(i % 7);
      break;
    case kTfLiteUInt8:
      for (int i = 0; i < total_size; ++i)
        input->data.uint8[i] = static_cast<uint8_t>(i % 7);
      break;
    default:
      MicroPrintf("Input fill skipped for unsupported tensor type (%d).", input->type);
      break;
  }
}

void LogOutput(const TfLiteTensor* output) {
  if (!g_print_outputs)
    return;

  const int total_size = CalculateElementCount(output);
  switch (output->type) {
    case kTfLiteFloat32:
      for (int i = 0; i < total_size; ++i)
        MicroPrintf("%f", static_cast<double>(output->data.f[i]));
      break;
    case kTfLiteInt8:
      for (int i = 0; i < total_size; ++i)
        MicroPrintf("%d", static_cast<int>(output->data.int8[i]));
      break;
    case kTfLiteUInt8:
      for (int i = 0; i < total_size; ++i)
        MicroPrintf("%u", static_cast<unsigned int>(output->data.uint8[i]));
      break;
    default:
      MicroPrintf("Output logging skipped for unsupported tensor type (%d).", output->type);
      break;
  }
}

template <int kMaxOps, typename AddOpsFn>
int InvokeModel(const unsigned char* model_data,
                size_t model_length,
                const char* model_name,
                AddOpsFn add_ops,
                uint8_t* tensor_arena,
                int tensor_arena_size,
                uint32_t (*get_time_ms)()) {
  MicroPrintf("%s: started (%u bytes).", model_name, static_cast<unsigned int>(model_length));
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  tflite::MicroMutableOpResolver<kMaxOps> op_resolver;
  TF_LITE_ENSURE_STATUS(add_ops(op_resolver));

#ifdef __arm__
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, tensor_arena_size);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
#else
  tflite::RecordingMicroInterpreter interpreter(model, op_resolver, tensor_arena, tensor_arena_size);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  interpreter.GetMicroAllocator().PrintAllocations();
#endif

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  FillInput(input);

  uint32_t start_ms = 0;
  uint32_t end_ms = 0;
  const bool timing_available = (get_time_ms != nullptr);

  if (timing_available)
    start_ms = get_time_ms();

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  if (timing_available) {
    end_ms = get_time_ms();
    const uint32_t duration_ms = end_ms - start_ms;
    MicroPrintf("%s: completed in %lu ms.", model_name, static_cast<unsigned long>(duration_ms));
  } else {
    MicroPrintf("%s: completed (timing unavailable).", model_name);
  }

  LogOutput(output);

  return 0;
}
}  // namespace

void register_debug_log_callback(void (*callback)(const char* s)) {
#ifdef __arm__
  RegisterDebugLogCallback(callback);
#endif
}

void set_print_output(int enable) {
  g_print_outputs = (enable != 0);
}

#define DEFINE_TFLM_MAIN(symbol, display_name)                                                      \
  int tflm_main_##symbol(uint8_t* tensor_arena, int tensor_arena_size, uint32_t (*get_time_ms)()) { \
    auto add_ops = [&](tflite::MicroMutableOpResolver<TFLM_MODEL_OP_COUNT_##symbol>& resolver) {    \
      TFLM_APPLY_MODEL_OPS_##symbol(resolver);                                                      \
      return kTfLiteOk;                                                                             \
    };                                                                                              \
    return InvokeModel<TFLM_MODEL_OP_COUNT_##symbol>(                                               \
      g_model_data_##symbol,                                                                        \
      g_model_data_##symbol##_len,                                                                  \
      display_name,                                                                                 \
      add_ops,                                                                                      \
      tensor_arena,                                                                                 \
      tensor_arena_size,                                                                            \
      get_time_ms                                                                                   \
    );                                                                                              \
  }
TFLM_FOREACH_MODEL(DEFINE_TFLM_MAIN)
#undef DEFINE_TFLM_MAIN

int tflm_main(uint8_t* tensor_arena, int tensor_arena_size, uint32_t (*get_time_ms)()) {
  bool had_failure = false;

#define RUN_MODEL(symbol, display_name)                                                  \
  {                                                                                      \
    const int status = tflm_main_##symbol(tensor_arena, tensor_arena_size, get_time_ms); \
    if (status != 0) {                                                                   \
      MicroPrintf("%s: failed (status %d).", display_name, status);                      \
      had_failure = true;                                                                \
    }                                                                                    \
    MicroPrintf("");                                                                     \
  }
TFLM_FOREACH_MODEL(RUN_MODEL);
#undef RUN_MODEL

  if (!had_failure)
    MicroPrintf("All models ran successfully.");
  else
    MicroPrintf("One or more models failed.");

  return had_failure ? 1 : 0;
}
