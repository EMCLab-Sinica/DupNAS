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




############################################################################
# MAIN COST MODEL - CONTINUOUS POWER
############################################################################
# get estimated total energy consumption in a power cycle
def est_cost_BN_contpow(layer, params_exec, plat_settings, plat_cost_profile):    
    #E_rb, L_rb = est_cost_BN_reboot_intpow(plat_cost_profile) # not included in model
    E_fd, L_fd = est_cost_BN_layerinputfetch_contpow(layer, params_exec, plat_cost_profile)    
    E_cp, L_cp = est_cost_BN_layercomp_contpow(layer, params_exec, plat_cost_profile)
    E_bd, L_bd = est_cost_BN_layeroutputbackup_contpow(layer, params_exec, plat_cost_profile)
    
    #E_fl, L_fl = est_cost_BN_tileidxfetch_intpow(plat_cost_profile)  # no tile indeces fetching/preserving in this model
    #E_bl, L_bl = est_cost_BN_tileidxbackup_intpow(plat_cost_profile) # no tile indeces fetching/preserving in this model

    cost_breakdown={
        "rb": [0, 0],
        "fd": [E_fd, L_fd],
        "fl": [0, 0],
        "cp": [E_cp, L_cp],
        "bd": [E_bd, L_bd],
        "bl": [0, 0]
    }

    total_energy = E_fd + E_cp + E_bd
    total_latency = L_fd + L_cp + L_bd

    return total_energy, total_latency, cost_breakdown

# get reboot energy cost
def est_cost_BN_reboot_contpow(plat_cost_profile):
    total_en_cost, total_lat_cost = est_cost_BN_reboot_intpow(plat_cost_profile)    
    return total_en_cost, total_lat_cost

# get layer data input fetch energy cost (overall layer)
def est_cost_BN_layerinputfetch_contpow(layer, params_exec, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0  
    
    # execution space    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']        
   
    # num of data transfer command invocations
    nI, nW, blkI, blkW = _num_datatrcmds_fetch_tile_data(params_exec)
        
    # energy/latency cost of the transfer (for a given block size)
    er_blkI = plat_cost_profile['E_DMA_NVM_TO_VM'](blkI); lr_blkI = plat_cost_profile['L_DMA_NVM_TO_VM'](blkI)    
    er_blkW = plat_cost_profile['E_DMA_NVM_TO_VM'](blkW); lr_blkW = plat_cost_profile['L_DMA_NVM_TO_VM'](blkW)    
    # energy/latency overhead of each transfer
    eofI = plat_cost_profile['E_FD_I_OVHD']; lofI = plat_cost_profile['L_FD_I_OVHD']    
    eofW = plat_cost_profile['E_FD_W_OVHD']; lofW = plat_cost_profile['L_FD_W_OVHD']    

    # -- calc energy cost (no reuse schemes)
    #tile_en_cost = (nI*(er_blkI + eofI)) + (nW*(er_blkW + eofW))
    #tile_lat_cost = (nI*(lr_blkI + lofI)) + (nW*(lr_blkW + lofW))
    
    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, layer_type=layer['type'])

    total_en_cost = (num_tiles * (nI*(er_blkI + eofI))) + (np.ceil(M/Tm) * (nW*(er_blkW + eofW)))
    total_lat_cost = (num_tiles * (nI*(lr_blkI + lofI))) + (np.ceil(M/Tm) * (nW*(lr_blkW + lofW)))
    
    return total_en_cost, total_lat_cost

# backup layer output energy cost (overall layer)
def est_cost_BN_layeroutputbackup_contpow(layer, params_exec, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0    

    # execution space
    inter_lo = params_exec['inter_lo']    # Unused, always reuse_W
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
          
    # num of data transfer command invocations
    nO, blkO = _num_datatrcmds_backup_tile_data(params_exec)
    # energy/latency cost of the transfer (for a given block size)        
    ew_blkO = plat_cost_profile['E_DMA_VM_TO_NVM'](blkO); lw_blkO = plat_cost_profile['L_DMA_VM_TO_NVM'](blkO)
    # energy/latency overhead of each transfer    
    eobO = plat_cost_profile['E_BD_O_RUW_OVHD']; lobO = plat_cost_profile['L_BD_O_RUW_OVHD']
    
    # -- calc energy cost depending on reuse scheme            
    tile_en_cost = nO * (ew_blkO + eobO) 
    tile_lat_cost = nO * (lw_blkO + lobO) 

    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, layer_type=layer['type'])

    total_en_cost = num_tiles * tile_en_cost
    total_lat_cost = num_tiles * tile_lat_cost    
        
    return total_en_cost, total_lat_cost

def est_cost_BN_layercomp_contpow(layer, params_exec, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0

    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']        
    
    # BN uses CPU
    eaddcomp = plat_cost_profile['E_ADD']    
    emulcomp = plat_cost_profile['E_MUL']    
    edivcomp = plat_cost_profile['E_DIV']    
    laddcomp = plat_cost_profile['L_ADD']    
    lmulcomp = plat_cost_profile['L_MUL']    
    ldivcomp = plat_cost_profile['L_DIV']    
        
    # emaxcomp = plat_cost_profile['E_OP_MAXCOMPARE']    
    # emaxcomp_ovh = plat_cost_profile['E_OP_MAX_OVHD'] # addressing overhead
    # lmaxcomp = plat_cost_profile['L_OP_MAXCOMPARE']    
    # lmaxcomp_ovh = plat_cost_profile['L_OP_MAX_OVHD'] # addressing overhead

    laddr_ovh = plat_cost_profile['L_OP_COMP_BN_OVHD']
    eaddr_ovh = plat_cost_profile['E_OP_COMP_BN_OVHD']
    
    tile_en_cost = Tr * Tc * (Tm * ((3*eaddcomp) + (2*emulcomp) +  eaddr_ovh))
    tile_lat_cost = Tr * Tc * (Tm * ((3*laddcomp) + (2*lmulcomp) + laddr_ovh))

    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, layer_type=layer['type'])
    
    #print("BN size: ", num_tiles, tile_lat_cost, tile_en_cost, H, W, R, C, M, N, Tr, Tc, Tm, Tn)
    #sys.exit()

    total_en_cost = num_tiles * tile_en_cost
    total_lat_cost = num_tiles * tile_lat_cost

    return total_en_cost, total_lat_cost




############################################################################
# MAIN COST MODEL - INTERMITTENT POWER (per power cycle)
############################################################################
# get estimated total energy consumption in a power cycle
def est_cost_BN_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile, return_only_breakdown=False):    
    E_rb, L_rb = est_cost_BN_reboot_intpow(plat_cost_profile)
    E_fd, L_fd = est_cost_BN_tileinputfetch_intpow(params_exec, params_pres, plat_cost_profile)
    E_fl, L_fl = est_cost_BN_tileidxfetch_intpow(plat_cost_profile)
    E_cp, L_cp = est_cost_BN_tilecomp_intpow(params_exec, params_pres, plat_cost_profile)
    E_bd, L_bd = est_cost_BN_tileoutputbackup_intpow(params_exec, params_pres, plat_cost_profile)
    E_bl, L_bl = est_cost_BN_tileidxbackup_intpow(plat_cost_profile)

    total_energy_powcycle = E_rb + E_fd + E_fl + E_cp + E_bd + E_bl
    total_latency_powcycle = L_rb + L_fd + L_fl + L_cp + L_bd + L_bl

    #pprint(params_exec); pprint(params_pres)
    #pprint([total_energy_powcycle, E_rb, E_fd, E_fl, E_cp, E_bd, E_bl]); 
    #sys.exit()
    
    cost_breakdown={
        "rb": [E_rb, L_rb],
        "fd": [E_fd, L_fd],
        "fl": [E_fl, L_fl],
        "cp": [E_cp, L_cp],
        "bd": [E_bd, L_bd],
        "bl": [E_bl, L_bl]
    }

    #pprint(cost_breakdown); sys.exit()
    
    if (return_only_breakdown==False):
        return total_energy_powcycle, total_latency_powcycle, cost_breakdown
    else:
        return E_rb, L_rb, E_fd, L_fd, E_fl, L_fl, E_cp, L_cp, E_bd, L_bd, E_bl, L_bl

# get reboot energy cost
def est_cost_BN_reboot_intpow(plat_cost_profile):
    total_en_cost = plat_cost_profile['E_RB']
    total_lat_cost = plat_cost_profile['L_RB']
    return total_en_cost, total_lat_cost

# get tile data input fetch energy cost per power cycle
def est_cost_BN_tileinputfetch_intpow(params_exec, params_pres, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0  
    
    # execution space    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # preservation space   
    S = params_pres['backup_batch_size']
   
    # num of data transfer command invocations
    nI, nW, blkI, blkW = _num_datatrcmds_fetch_tile_data(params_exec)
        
    # energy/latency cost of the transfer (for a given block size)
    er_blkI = plat_cost_profile['E_DMA_NVM_TO_VM'](blkI); lr_blkI = plat_cost_profile['L_DMA_NVM_TO_VM'](blkI)    
    er_blkW = plat_cost_profile['E_DMA_NVM_TO_VM'](blkW); lr_blkW = plat_cost_profile['L_DMA_NVM_TO_VM'](blkW)    
    # energy/latency overhead of each transfer
    eofI = plat_cost_profile['E_FD_I_OVHD']; lofI = plat_cost_profile['L_FD_I_OVHD']    
    eofW = plat_cost_profile['E_FD_W_OVHD']; lofW = plat_cost_profile['L_FD_W_OVHD']    

    # -- calc energy cost (only reuse_W is helpful for BatchNorm)
    total_en_cost = (S * (nI*(er_blkI + eofI)) + (nW*(er_blkW + eofW)))
    total_lat_cost = (S * (nI*(lr_blkI + lofI)) + (nW*(lr_blkW + lofW)))
    
    return total_en_cost, total_lat_cost
    
# get tile data output backup energy cost per power cycle
def est_cost_BN_tileoutputbackup_intpow(params_exec, params_pres, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0    

    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # preservation space   
    S = params_pres['backup_batch_size']
   
    # num of data transfer command invocations
    nO, blkO = _num_datatrcmds_backup_tile_data(params_exec)
    # energy/latency cost of the transfer (for a given block size)        
    ew_blkO = plat_cost_profile['E_DMA_VM_TO_NVM'](blkO); lw_blkO = plat_cost_profile['L_DMA_VM_TO_NVM'](blkO)  # preservation batch size in the Tr, Tc direction, so output buffer is overwritten
    # energy/latency overhead of each transfer    
    eobO = plat_cost_profile['E_BD_O_RUW_OVHD']; lobO = plat_cost_profile['L_BD_O_RUW_OVHD']
    
    # -- calc energy cost depending on reuse scheme            
    total_en_cost = nO * (ew_blkO + eobO) 
    total_lat_cost = nO * (lw_blkO + lobO) 
        
    return total_en_cost, total_lat_cost

def est_cost_BN_tilecomp_intpow(params_exec, params_pres, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0

    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # preservation space   
    S = params_pres['backup_batch_size']
   
    # BN uses CPU
    eaddcomp = plat_cost_profile['E_ADD']    
    emulcomp = plat_cost_profile['E_MUL']    
    edivcomp = plat_cost_profile['E_DIV']    
    laddcomp = plat_cost_profile['L_ADD']    
    lmulcomp = plat_cost_profile['L_MUL']    
    ldivcomp = plat_cost_profile['L_DIV']    
    
    # emaxcomp = plat_cost_profile['E_OP_MAXCOMPARE']    
    # emaxcomp_ovh = plat_cost_profile['E_OP_MAX_OVHD'] # addressing overhead
    # lmaxcomp = plat_cost_profile['L_OP_MAXCOMPARE']    
    # lmaxcomp_ovh = plat_cost_profile['L_OP_MAX_OVHD'] # addressing overhead
    
    #total_en_cost = S * Kh * Kw * Tr * Tc * Tm * (emaxcomp + emaxcomp_ovh)
    #total_lat_cost = S * Kh * Kw * Tr * Tc * Tm * (lmaxcomp + lmaxcomp_ovh)

    laddr_ovh = plat_cost_profile['L_OP_COMP_BN_OVHD']
    eaddr_ovh = plat_cost_profile['E_OP_COMP_BN_OVHD']

    total_en_cost = S * Tr * Tc * (Tm * ((3*eaddcomp) + (2*emulcomp) + eaddr_ovh))
    total_lat_cost = S * Tr * Tc * (Tm * ((2*laddcomp) + (2*lmulcomp) + laddr_ovh))

    return total_en_cost, total_lat_cost

# cost of fetch inter tile indices
def est_cost_BN_tileidxfetch_intpow(plat_cost_profile):
    total_en_cost = plat_cost_profile['E_DMA_NVM_TO_VM'](5) # fetch 4 but use only 3? [4 tile indices + 1 layer index]
    total_lat_cost = plat_cost_profile['L_DMA_NVM_TO_VM'](5)
    return total_en_cost, total_lat_cost

# cost of backup inter tile indices
def est_cost_BN_tileidxbackup_intpow(plat_cost_profile):
    total_en_cost = plat_cost_profile['E_DMA_VM_TO_NVM'](5) # backup 4 but use only 3? [4 tile indices + 1 layer index]
    total_lat_cost = plat_cost_profile['L_DMA_VM_TO_NVM'](5)
    return total_en_cost, total_lat_cost



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








    

















