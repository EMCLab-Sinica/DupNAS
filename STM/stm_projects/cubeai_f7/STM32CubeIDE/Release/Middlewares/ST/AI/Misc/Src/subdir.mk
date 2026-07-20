################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.c \
REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/aiTestUtility.c \
REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/ai_device_adaptor.c \
REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/lc_print.c \
REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/syscalls.c 

OBJS += \
./Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.o \
./Middlewares/ST/AI/Misc/Src/aiTestUtility.o \
./Middlewares/ST/AI/Misc/Src/ai_device_adaptor.o \
./Middlewares/ST/AI/Misc/Src/lc_print.o \
./Middlewares/ST/AI/Misc/Src/syscalls.o 

C_DEPS += \
./Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.d \
./Middlewares/ST/AI/Misc/Src/aiTestUtility.d \
./Middlewares/ST/AI/Misc/Src/ai_device_adaptor.d \
./Middlewares/ST/AI/Misc/Src/lc_print.d \
./Middlewares/ST/AI/Misc/Src/syscalls.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.o: REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.c Middlewares/ST/AI/Misc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/ST/AI/Misc/Src/aiTestUtility.o: REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/aiTestUtility.c Middlewares/ST/AI/Misc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/ST/AI/Misc/Src/ai_device_adaptor.o: REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/ai_device_adaptor.c Middlewares/ST/AI/Misc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/ST/AI/Misc/Src/lc_print.o: REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/lc_print.c Middlewares/ST/AI/Misc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/ST/AI/Misc/Src/syscalls.o: REPO_ROOT/stm_projects/cubeai_f7/Middlewares/ST/AI/Misc/Src/syscalls.c Middlewares/ST/AI/Misc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-ST-2f-AI-2f-Misc-2f-Src

clean-Middlewares-2f-ST-2f-AI-2f-Misc-2f-Src:
	-$(RM) ./Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.cyclo ./Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.d ./Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.o ./Middlewares/ST/AI/Misc/Src/aiTestHelper_ST_AI.su ./Middlewares/ST/AI/Misc/Src/aiTestUtility.cyclo ./Middlewares/ST/AI/Misc/Src/aiTestUtility.d ./Middlewares/ST/AI/Misc/Src/aiTestUtility.o ./Middlewares/ST/AI/Misc/Src/aiTestUtility.su ./Middlewares/ST/AI/Misc/Src/ai_device_adaptor.cyclo ./Middlewares/ST/AI/Misc/Src/ai_device_adaptor.d ./Middlewares/ST/AI/Misc/Src/ai_device_adaptor.o ./Middlewares/ST/AI/Misc/Src/ai_device_adaptor.su ./Middlewares/ST/AI/Misc/Src/lc_print.cyclo ./Middlewares/ST/AI/Misc/Src/lc_print.d ./Middlewares/ST/AI/Misc/Src/lc_print.o ./Middlewares/ST/AI/Misc/Src/lc_print.su ./Middlewares/ST/AI/Misc/Src/syscalls.cyclo ./Middlewares/ST/AI/Misc/Src/syscalls.d ./Middlewares/ST/AI/Misc/Src/syscalls.o ./Middlewares/ST/AI/Misc/Src/syscalls.su

.PHONY: clean-Middlewares-2f-ST-2f-AI-2f-Misc-2f-Src

