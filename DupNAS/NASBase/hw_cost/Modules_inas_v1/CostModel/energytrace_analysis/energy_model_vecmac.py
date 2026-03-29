import sys, os



###############################
# Model params
###############################
#PARAM_LEAVECMAC_LINREG_M = 5.27308646249e-10
#PARAM_LEAVECMAC_LINREG_B = 2.04412689725e-07
PARAM_LEAVECMAC_LINREG_M = 5.03343396188007e-10
PARAM_LEAVECMAC_LINREG_B = 2.057298797039179e-07
   


######################################################
#   Modelling
######################################################
# linear prediction
def predict_leavecmac_energy(tr_type, vec_size):
    if (tr_type == "LEAVECMAC"):        
        y = (PARAM_LEAVECMAC_LINREG_M * vec_size) + PARAM_LEAVECMAC_LINREG_B    
    else:
        sys.exit("Error: predict_dma_energy: unknown")
    return y

