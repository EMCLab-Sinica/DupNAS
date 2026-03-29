import sys, os




###############################
# Model params
###############################
PARAM_FRAM_TO_SRAM_ENERGY = 1.0261111111111113e-08
PARAM_SRAM_TO_FRAM_ENERGY = 9.602777777777778e-09
PARAM_FRAM_TO_FRAM_ENERGY = 1.2233055555555557e-08
PARAM_SRAM_TO_SRAM_ENERGY = 3.88611111111111e-09



######################################################
#   Modelling
######################################################
# linear prediction
def predict_cpu_data_energy(tr_type, data_size):

    if (tr_type == "FRAM_TO_SRAM"):
        return PARAM_FRAM_TO_SRAM_ENERGY
    elif (tr_type == "SRAM_TO_FRAM"):
        return PARAM_SRAM_TO_FRAM_ENERGY
    elif (tr_type == "FRAM_TO_FRAM"):
        return PARAM_FRAM_TO_FRAM_ENERGY
    elif (tr_type == "SRAM_TO_SRAM"):
        return PARAM_SRAM_TO_SRAM_ENERGY
    else:
        sys.exit("Error: predict_cpu_data_energy: unknown")


