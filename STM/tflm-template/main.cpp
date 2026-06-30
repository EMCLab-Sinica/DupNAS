#include <chrono>
#include <cstdlib>
#include "src/tflm_main.h"

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static uint32_t HostGetTimeMs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

int main(int argc, char** argv) {
    set_print_output(argc > 1 && std::atoi(argv[1]) != 0);
    return tflm_main(tensor_arena, TENSOR_ARENA_SIZE, HostGetTimeMs);
}
