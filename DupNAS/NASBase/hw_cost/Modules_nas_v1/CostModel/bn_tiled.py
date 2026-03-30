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
    blkI = Tm            
    blkW = Tm
    # num of transfers per buffer type
    nI = Tri * Tci        
    nW = 4   # mean, stdandard deviation, beta, gamma
    return nI, nW, blkI, blkW

def _num_datatrcmds_backup_tile_data(params_exec, dma_type='1D'):
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # block size for each DMA transfer per buffer type    
    blkO = Tm
    # num of transfers per buffer type    
    nO = Tr * Tc
    return nO, blkO









def est_cost_BN_flops(layer, params_exec, params_pres, layer_based_cals):
    # execution, preservation space params
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    inter_lo = params_exec['inter_lo']    
    S = params_pres['backup_batch_size']

    if layer_based_cals:
        total_macs = 0  # No macs in BN
        total_flops = R * C * M * 4  # one SUB, one DIV, one MUL, one ADD for each input feature
        return total_flops, total_macs
    
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn)    
        
    # TrTcTm MAC+ TrTcTm ADD    
        
    total_macs = (S * Tr * Tc * Tm) * num_tiles
    total_flops = ((S * Tr * Tc * Tm * 2) + (S * Tr * Tc * Tm)) * num_tiles
    
    return total_flops, total_macs








    

















