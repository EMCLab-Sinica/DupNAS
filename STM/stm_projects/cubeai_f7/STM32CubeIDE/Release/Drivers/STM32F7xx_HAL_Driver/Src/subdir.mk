################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
REPO_ROOT/stm_projects/cubeai_f7/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.c \
REPO_ROOT/stm_projects/cubeai_f7/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.c 

OBJS += \
./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.o \
./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.o 

C_DEPS += \
./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.d \
./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.o: REPO_ROOT/stm_projects/cubeai_f7/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.c Drivers/STM32F7xx_HAL_Driver/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.o: REPO_ROOT/stm_projects/cubeai_f7/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.c Drivers/STM32F7xx_HAL_Driver/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -DHAVE_NETWORK_INFO -c -I../../Core/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../Drivers/CMSIS/Include -I../../Middlewares/ST/AI/Inc -I../../Middlewares/ST/AI/Misc/Inc -I../../Middlewares/ST/AI/Misc/Src -I../../Drivers/BSP/STM32746G-Discovery -I../../Drivers/STM32F7xx_HAL_Driver/Src -I../../AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-STM32F7xx_HAL_Driver-2f-Src

clean-Drivers-2f-STM32F7xx_HAL_Driver-2f-Src:
	-$(RM) ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.cyclo ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.d ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.o ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.su ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.cyclo ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.d ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.o ./Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.su

.PHONY: clean-Drivers-2f-STM32F7xx_HAL_Driver-2f-Src

