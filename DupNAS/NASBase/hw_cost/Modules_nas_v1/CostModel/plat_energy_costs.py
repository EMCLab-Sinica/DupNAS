import sys

from .energytrace_analysis.energy_model_dma import predict_dma_energy
from .energytrace_analysis.energy_model_dma import predict_dma_energy
from .energytrace_analysis.energy_model_vecmac import predict_leavecmac_energy
from .energytrace_analysis.energy_model_mathops import predict_mathop_energy
from .latency_analysis.latency_model_microbench import predict_latency


CPU_CLOCK_MSP430 = 16000000
NVM_SPEED_SCALE_FACTOR = 1   

SPI_CLOCK_DIVIDER = 4
SPI_CLOCK = CPU_CLOCK_MSP430 / SPI_CLOCK_DIVIDER  # check SPIBRW in inference library codes


# if we use DATA_SZ=1, do these latencies and energy levels need to be scaled down (halved) ?
DATA_SZ_SCALE_FACTOR_LATENCY = 1    # original cost model was profiled on 16 bit data size, set appropriately to scale up/down latency/energy model
DATA_SZ_SCALE_FACTOR_ENERGY = 1    # original cost model was profiled on 16 bit data size, set appropriately to scale up/down latency/energy model

TMP_MIN_E = 0.000001
TMP_MIN_L = 0.001



# proxy functions - pointing to real platform measurements ---

def _proxy_datatr_rd_energy(v):
    return predict_dma_energy("FRAM_TO_SRAM", v * DATA_SZ_SCALE_FACTOR_ENERGY) * NVM_SPEED_SCALE_FACTOR 
def _proxy_datatr_wr_energy(v):
    return predict_dma_energy("SRAM_TO_FRAM", v * DATA_SZ_SCALE_FACTOR_ENERGY) * NVM_SPEED_SCALE_FACTOR 
def _proxy_leavecmac_energy(v):
    return predict_leavecmac_energy("LEAVECMAC", v)

E_ADD = predict_mathop_energy("ADD") * 1.0
E_MUL = predict_mathop_energy("MULTIPLY") * 1.0
E_DIV = predict_mathop_energy("DIVIDE") * 1.0
E_MOD = predict_mathop_energy("MODULO") * 1.0
E_MAX = predict_mathop_energy("MAX") * 1.0


# latency is in clock cycles, need to convert to seconds

L_ADD = predict_latency("MATHOPS_ADD", None) / CPU_CLOCK_MSP430
L_MUL = predict_latency("MATHOPS_MUL", None) / CPU_CLOCK_MSP430
L_DIV = predict_latency("MATHOPS_DIV", None) / CPU_CLOCK_MSP430
L_MOD = predict_latency("MATHOPS_MOD", None) / CPU_CLOCK_MSP430
L_MAX = predict_latency("MATHOPS_MAX", None) / CPU_CLOCK_MSP430

def _proxy_datatr_rd_latency(v):
    return ((predict_latency("FRAM_TO_SRAM", v * DATA_SZ_SCALE_FACTOR_LATENCY) / SPI_CLOCK) * NVM_SPEED_SCALE_FACTOR ) 
def _proxy_datatr_wr_latency(v):
    return ((predict_latency("SRAM_TO_FRAM", v * DATA_SZ_SCALE_FACTOR_LATENCY) / SPI_CLOCK) * NVM_SPEED_SCALE_FACTOR )
def _proxy_leavecmac_latency(v):
    return (predict_latency("LEAVECMAC", v) / CPU_CLOCK_MSP430) 



# -- temp functions used for testing -- 
# def _temp_datatr_rd_energy(v):
#     return v*TMP_MIN_E*8
# def _temp_datatr_wr_energy(v):
#     return v*TMP_MIN_E*10

# def _temp_datatr_latency(v):
#     return v*TMP_MIN_L*5
# def _temp_comp_energy(v):
#     return v*TMP_MIN_E
# def _temp_comp_latency(v):
#     return v*TMP_MIN_L



class PlatformCostModel:
    

    PLAT_MSP430_EXTNVM = {
        
        # =========== Energy (in Joules) ===========        
        # reboot cost
        "E_RB" : 0.00007788,  # E=0.5 * C * (V1^2 - V2^2) = 0.5 * 0.00033 * (2.99^2 - 2.91^2)        
                 
        # CPU ops cost
        "E_ADD" : E_ADD,
        "E_SUB" : E_ADD, # assume same cost as add
        "E_MUL" : E_MUL,
        "E_DIV" : E_DIV,
        "E_MOD" : E_MOD,

        # data move cost
        "E_DMA_NVM_TO_VM" : _proxy_datatr_rd_energy,
        "E_DMA_VM_TO_NVM" : _proxy_datatr_wr_energy,        
        # data fetch overhead (addressing)
        # "E_FD_I_OVHD" : (E_ADD * (5+2)) + (E_MUL * (3+2)),            # according to asm: (E_ADD * (8+2)) + (E_MUL * (3+2)),
        # "E_FD_W_OVHD" : (E_ADD * (5+3)) + (E_MUL * (6+4)),            # according to asm: (E_ADD * (7+3)) + (E_MUL * (6+3)),
        # "E_FD_O_OVHD" : (E_ADD * (5+2)) + (E_MUL * (3+2)),            # according to asm: (E_ADD * (6+2)) + (E_MUL * (2+2)),
        # "E_BD_O_OVHD" : (E_ADD * (2+5)) + (E_MUL * (3+7)),            # according to asm: (E_ADD * (2+8)) + (E_MUL * (3+6)),
        "E_FD_I_OVHD" : (E_ADD * (8)) + (E_MUL * (4)),            
        "E_FD_W_OVHD" : (E_ADD * (9)) + (E_MUL * (9)),                    
        "E_FD_O_RUI_OVHD" : (E_ADD * 8) + (E_MUL * 5),            
        "E_FD_O_RUW_OVHD" : (E_ADD * 8) + (E_MUL * 4),            
        "E_FD_O_RUO_OVHD" : (E_ADD * 8) + (E_MUL * 4),            
        "E_BD_O_RUI_OVHD" : (E_ADD * 8) + (E_MUL * 8),            
        "E_BD_O_RUW_OVHD" : (E_ADD * 8) + (E_MUL * 9),            
        "E_BD_O_RUO_OVHD" : (E_ADD * 8) + (E_MUL * 5),            


        # vector MAC compute cost
        "E_OP_VECMAC" : _proxy_leavecmac_energy,
        # computation invocation overhead (addressing)
        "E_OP_PSUM_OVHD" : (E_ADD * (4+3)) + (E_MUL * (2+3)),               # according to asm: (E_ADD * (4+3)) + (E_MUL * (2+3)),  
        "E_OP_ACUM_OVHD" : (E_ADD * (3)) + (E_MUL * (3)),                   # according to asm: (E_ADD * (3)) + (E_MUL * (3)),  

        # updated for 
        "E_OP_COMP_ADD_OVHD" : (E_ADD * 4) + (E_MUL * 2),
        "E_OP_COMP_BN_OVHD" : (E_ADD * 4) + (E_MUL * 2),                    # according to asm: (E_ADD * 7) + (E_MUL * 2),
        # combined psum+acum (cost slightly higher for IFM reuse, so we take that case)
        "E_OP_COMP_CONV_OVHD" : (E_ADD * 12) + (E_MUL * 9),
        "E_OP_COMP_DWCONV_OVHD_IFM" : (E_ADD * 11) + (E_MUL * 12),
        "E_OP_COMP_DWCONV_OVHD_OTHER" : (E_ADD * 11) + (E_MUL * 10),


        # q15 compare operation
        "E_OP_MAXCOMPARE" : E_MAX,
        "E_OP_MAX_OVHD"   : (E_ADD * (2)) + (E_MUL * (1)),                  # according to asm: (E_ADD * (2)) + (E_MUL * (1)),  


        # =========== Latency (in seconds) ===========
        # reboot cost
        "L_RB" : 0.055, # 55 ms
        #"L_RB" : 0.07, # 70 ms
        #"L_RB" : 0.02, # 20 ms
        #"L_RB" : 0.028, # 28 ms

        # CPU ops cost
        "L_ADD" : L_ADD,
        "L_SUB" : L_ADD, # assume same cost as add
        "L_MUL" : L_MUL,
        "L_DIV" : L_DIV,
        "L_MOD" : L_MOD,

        # data move cost
        "L_DMA_NVM_TO_VM" : _proxy_datatr_rd_latency,
        "L_DMA_VM_TO_NVM" : _proxy_datatr_wr_latency,
        # data fetch overhead (addressing) - after checking assembly
        "L_FD_I_OVHD" : (L_ADD * (8)) + (L_MUL * (4)),            
        "L_FD_W_OVHD" : (L_ADD * (9)) + (L_MUL * (9)),                    
        "L_FD_O_RUI_OVHD" : (L_ADD * 8) + (L_MUL * 5),            
        "L_FD_O_RUW_OVHD" : (L_ADD * 8) + (L_MUL * 4),            
        "L_FD_O_RUO_OVHD" : (L_ADD * 8) + (L_MUL * 4),            
        "L_BD_O_RUI_OVHD" : (L_ADD * 8) + (L_MUL * 8),            
        "L_BD_O_RUW_OVHD" : (L_ADD * 8) + (L_MUL * 9),            
        "L_BD_O_RUO_OVHD" : (L_ADD * 8) + (L_MUL * 5),            


        # vector MAC compute cost
        "L_OP_VECMAC" : _proxy_leavecmac_latency,
        # computation invocation overhead (addressing)
        "L_OP_PSUM_OVHD" : (L_ADD * (4+3)) + (L_MUL * (4+6)),
        "L_OP_ACUM_OVHD" : (L_ADD * (3)) + (L_MUL * (3)),
        
        # updated for 
        "L_OP_COMP_ADD_OVHD" : (L_ADD * 4) + (L_MUL * 2),
        "L_OP_COMP_BN_OVHD" : (L_ADD * 4) + (L_MUL * 2),                    # according to asm: (E_ADD * 7) + (E_MUL * 2),
        # combined psum+acum (cost slightly higher for IFM reuse, so we take that case)
        "L_OP_COMP_CONV_OVHD" : (L_ADD * 12) + (L_MUL * 9),
        "L_OP_COMP_DWCONV_OVHD_IFM" : (L_ADD * 11) + (L_MUL * 12),
        "L_OP_COMP_DWCONV_OVHD_OTHER" : (L_ADD * 11) + (L_MUL * 10),

        # q15 compare operation
        "L_OP_MAXCOMPARE" : L_MAX,
        "L_OP_MAX_OVHD"   : (L_ADD * (2)) + (L_MUL * (1)),

    }



    PLAT_MSP432_EXTNVM = {
        
        "test" : 1        
        
    }



