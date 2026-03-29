import sys, os



###############################
# Model params
###############################
#PARAM_ADD_ENERGY = 3.935538461538471e-09
#PARAM_MULTIPLY_ENERGY = 1.9253346153846146e-08
#PARAM_DIVIDE_ENERGY = 4.373834615384616e-08
#PARAM_MODULO_ENERGY = 4.366230769230769e-08
#PARAM_MAX_ENERGY = 4.228153846153846e-09
PARAM_ADD_ENERGY = 3.4965641025641274e-09
PARAM_MULTIPLY_ENERGY = 1.892907692307692e-08
PARAM_DIVIDE_ENERGY = 4.1691282051282045e-08
PARAM_MODULO_ENERGY = 4.158656410256406e-08
PARAM_MAX_ENERGY = 3.9232307692307644e-09


######################################################
#   Modelling
######################################################
# linear prediction
def predict_mathop_energy(tr_type):

    if (tr_type == "ADD"):
        return PARAM_ADD_ENERGY
    elif (tr_type == "MULTIPLY"):
        return PARAM_MULTIPLY_ENERGY
    elif (tr_type == "DIVIDE"):
        return PARAM_DIVIDE_ENERGY
    elif (tr_type == "MODULO"):
        return PARAM_MODULO_ENERGY
    elif (tr_type == "MAX"):
        return PARAM_MAX_ENERGY
    else:
        sys.exit("Error: predict_mathop_energy: unknown")





