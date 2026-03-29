import sys, os


###############################
# Model params
###############################

# PARAM_FRAM_TO_SRAM_LINREG_M = 2.7052040637907215e-08
# PARAM_FRAM_TO_SRAM_LINREG_B = 2.0379416010498598e-07
# PARAM_SRAM_TO_FRAM_LINREG_M = 2.6815314305733587e-08
# PARAM_SRAM_TO_FRAM_LINREG_B = 1.9336196686351718e-07

# PARAM_FRAM_TO_SRAM_LINREG_M = 2.7e-08
# PARAM_FRAM_TO_SRAM_LINREG_B = 2.0e-07
# PARAM_SRAM_TO_FRAM_LINREG_M = 2.6815e-08
# PARAM_SRAM_TO_FRAM_LINREG_B = 1.9336e-07

# Profile updated 2024/01/27
PARAM_FRAM_TO_SRAM_LINREG_M= 2.4096781407047943e-08
PARAM_FRAM_TO_SRAM_LINREG_B= 1.9269110618985066e-07
PARAM_SRAM_TO_FRAM_LINREG_M= 2.3866387540946922e-08
PARAM_SRAM_TO_FRAM_LINREG_B= 1.9880671888670114e-07

######################################################
#   Modelling
######################################################
# linear prediction
def predict_dma_energy(tr_type, data_size):

    if (tr_type == "FRAM_TO_SRAM"):        
        y = (PARAM_FRAM_TO_SRAM_LINREG_M * data_size) + PARAM_FRAM_TO_SRAM_LINREG_B    
    
    elif (tr_type == "SRAM_TO_FRAM"):
        y = (PARAM_SRAM_TO_FRAM_LINREG_M * data_size) + PARAM_SRAM_TO_FRAM_LINREG_B

    else:
        sys.exit("Error: predict_dma_energy: unknown")

    return y



