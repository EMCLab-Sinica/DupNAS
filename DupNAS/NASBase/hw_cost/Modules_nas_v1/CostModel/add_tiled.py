import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect


# local imports
from . import common



############################################################################
# HELPERS
############################################################################

# assuming a 1D DMA transfer (non-strided)
def _num_datatrcmds_fetch_tile_data(params_exec, dma_type='1D'):
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # block size for each DMA transfer per buffer type
    # two operands summed up together
    blkI_1 = Tm            # Tm = Tn
    blkI_2 = Tm                
    # num of transfers per buffer type
    nI_1 = Tri * Tci       
    nI_2 = Tri * Tci       
    return nI_1, blkI_1, nI_2, blkI_2

def _num_datatrcmds_backup_tile_data(params_exec, dma_type='1D'):
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # block size for each DMA transfer per buffer type    
    blkO = Tm
    # num of transfers per buffer type    
    nO = Tr * Tc
    return nO, blkO






def est_cost_ADD_flops(layer, params_exec, params_pres, layer_based_cals):
    # execution, preservation space params
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    inter_lo = params_exec['inter_lo']    
    S = params_pres['backup_batch_size']

    if layer_based_cals:
        total_flops = R * C * M
        total_macs = 0
        return total_flops, total_macs
    
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn)    
        
    # TrTcTm MAC+ TrTcTm ADD    
        
    total_macs = 0
    total_flops = (S * Tr * Tc * Tm) * num_tiles
    
    return total_flops, total_macs








    

















