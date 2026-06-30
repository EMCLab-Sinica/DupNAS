################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/app_x-cube-ai.c \
REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/network.c \
REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/network_data.c \
REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/network_weights.c \
REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/user_init.c 

OBJS += \
./CM7/AI/App/app_x-cube-ai.o \
./CM7/AI/App/network.o \
./CM7/AI/App/network_data.o \
./CM7/AI/App/network_weights.o \
./CM7/AI/App/user_init.o 

C_DEPS += \
./CM7/AI/App/app_x-cube-ai.d \
./CM7/AI/App/network.d \
./CM7/AI/App/network_data.d \
./CM7/AI/App/network_weights.d \
./CM7/AI/App/user_init.d 


# Each subdirectory must supply rules for building sources it contributes
CM7/AI/App/app_x-cube-ai.o: REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/app_x-cube-ai.c CM7/AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DCORE_CM7 -DUSE_HAL_DRIVER -DSTM32H747xx -DUSE_PWR_DIRECT_SMPS_SUPPLY -DHAVE_NETWORK_INFO -c -I../../../CM7/Core/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -I../../../Drivers/CMSIS/Device/ST/STM32H7xx/Include -I../../../Drivers/CMSIS/Include -I../../../Middlewares/ST/AI/Inc -I../../../Middlewares/ST/AI/Misc/Inc -I../../../Middlewares/ST/AI/Misc/Src -I../../../CM7/AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
CM7/AI/App/network.o: REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/network.c CM7/AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DCORE_CM7 -DUSE_HAL_DRIVER -DSTM32H747xx -DUSE_PWR_DIRECT_SMPS_SUPPLY -DHAVE_NETWORK_INFO -c -I../../../CM7/Core/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -I../../../Drivers/CMSIS/Device/ST/STM32H7xx/Include -I../../../Drivers/CMSIS/Include -I../../../Middlewares/ST/AI/Inc -I../../../Middlewares/ST/AI/Misc/Inc -I../../../Middlewares/ST/AI/Misc/Src -I../../../CM7/AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
CM7/AI/App/network_data.o: REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/network_data.c CM7/AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DCORE_CM7 -DUSE_HAL_DRIVER -DSTM32H747xx -DUSE_PWR_DIRECT_SMPS_SUPPLY -DHAVE_NETWORK_INFO -c -I../../../CM7/Core/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -I../../../Drivers/CMSIS/Device/ST/STM32H7xx/Include -I../../../Drivers/CMSIS/Include -I../../../Middlewares/ST/AI/Inc -I../../../Middlewares/ST/AI/Misc/Inc -I../../../Middlewares/ST/AI/Misc/Src -I../../../CM7/AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
CM7/AI/App/network_weights.o: REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/network_weights.c CM7/AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DCORE_CM7 -DUSE_HAL_DRIVER -DSTM32H747xx -DUSE_PWR_DIRECT_SMPS_SUPPLY -DHAVE_NETWORK_INFO -c -I../../../CM7/Core/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -I../../../Drivers/CMSIS/Device/ST/STM32H7xx/Include -I../../../Drivers/CMSIS/Include -I../../../Middlewares/ST/AI/Inc -I../../../Middlewares/ST/AI/Misc/Inc -I../../../Middlewares/ST/AI/Misc/Src -I../../../CM7/AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
CM7/AI/App/user_init.o: REPO_ROOT/stm_projects/cubeai_h7/CM7/AI/App/user_init.c CM7/AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DCORE_CM7 -DUSE_HAL_DRIVER -DSTM32H747xx -DUSE_PWR_DIRECT_SMPS_SUPPLY -DHAVE_NETWORK_INFO -c -I../../../CM7/Core/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc -I../../../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -I../../../Drivers/CMSIS/Device/ST/STM32H7xx/Include -I../../../Drivers/CMSIS/Include -I../../../Middlewares/ST/AI/Inc -I../../../Middlewares/ST/AI/Misc/Inc -I../../../Middlewares/ST/AI/Misc/Src -I../../../CM7/AI/App -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-CM7-2f-AI-2f-App

clean-CM7-2f-AI-2f-App:
	-$(RM) ./CM7/AI/App/app_x-cube-ai.cyclo ./CM7/AI/App/app_x-cube-ai.d ./CM7/AI/App/app_x-cube-ai.o ./CM7/AI/App/app_x-cube-ai.su ./CM7/AI/App/network.cyclo ./CM7/AI/App/network.d ./CM7/AI/App/network.o ./CM7/AI/App/network.su ./CM7/AI/App/network_data.cyclo ./CM7/AI/App/network_data.d ./CM7/AI/App/network_data.o ./CM7/AI/App/network_data.su ./CM7/AI/App/network_weights.cyclo ./CM7/AI/App/network_weights.d ./CM7/AI/App/network_weights.o ./CM7/AI/App/network_weights.su ./CM7/AI/App/user_init.cyclo ./CM7/AI/App/user_init.d ./CM7/AI/App/user_init.o ./CM7/AI/App/user_init.su

.PHONY: clean-CM7-2f-AI-2f-App

