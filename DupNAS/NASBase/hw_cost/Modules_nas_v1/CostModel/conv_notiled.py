import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect


# local imports
from . import common
from ....model.common_types import OPTYPES

DMA_BUF = 32

############################################################################
# HELPERS
############################################################################

# assuming a 1D DMA transfer (non-strided)
def _num_datatrcmds_fetch_notile_data(layer, dma_type='1D'):
    #Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    H, W, R, C, M, N, Kh, Kw, stride = get_layer_props(layer)
    
    
    if layer['optype'] in (OPTYPES.O_CONV1D_DW, OPTYPES.O_CONV2D_DW):
        # block size for each DMA transfer per buffer type
        # Always use Tm as Tn=1
        blkI = DMA_BUF
        blkW = 1  # Always fetch one channel only, as each channel is fetched separately
        blkO = DMA_BUF

        # num of transfers per buffer type
        nI = np.ceil(H*W*N /DMA_BUF)
        nW = Kh * Kw * M
        nO = np.ceil(R*C*M /DMA_BUF)
    elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
        
        # block size for each DMA transfer per buffer type
        blkI =  DMA_BUF
        blkW =  1
        blkO =  DMA_BUF

        # num of transfers per buffer type
        nI = np.ceil(H*W*N /DMA_BUF)
        nW = Kh*Kw*M*N
        nO = np.ceil(R*C*M /DMA_BUF)

    else:    
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown op_type: " + OPTYPES.get_optype_label(layer['optype']))

    return nI, nW, nO, blkI, blkW, blkO

def _num_datatrcmds_backup_tile_data(layer, params_exec, dma_type='1D'):
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # block size for each DMA transfer per buffer type    
    blkO = Tm

    # num of transfers per buffer type    
    nO = Tr * Tc

    return nO, blkO



############################################################################
# MAIN COST MODEL
############################################################################


# end to end for whole layer
def est_cost_CONV_flops(layer, params_exec, params_pres, layer_based_cals):
    # execution, preservation space params
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    inter_lo = params_exec['inter_lo']    
    S = params_pres['backup_batch_size']

    if layer_based_cals:
        if layer['optype'] in (OPTYPES.O_CONV2D_DW, OPTYPES.O_CONV1D_DW):
            total_macs = Kh * Kw * R * C * M
            total_flops = 2 * total_macs  # XXX: does not match tile based results. Which is correct?

        elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
            total_macs = Kh * Kw * R * C * M * N
            total_flops = 2 * total_macs

        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown op_type")

        return total_flops, total_macs
    
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, op_type=layer['optype'])    
    
    if layer['optype'] in (OPTYPES.O_CONV2D_DW, OPTYPES.O_CONV1D_DW):
        total_macs = (S * Kh * Kw * Tr * Tc * Tm) * num_tiles
        total_flops = (S * Tr * Tc * Tm * ((Kh * Kw)+1)) * num_tiles

    elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
        total_macs = (S * Kh * Kw * Tr * Tc * Tm * Tn) * num_tiles
        total_flops = (S * Kh * Kw * Tr * Tc * Tm * ((2 * Tn) + 1)) * num_tiles    
        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown op_type")
    
    
    return total_flops, total_macs
    









    

















