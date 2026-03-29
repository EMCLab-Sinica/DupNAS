import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect


# local imports
from . import common
from ....model.common_types import OPTYPES


############################################################################
# HELPERS
############################################################################

# assuming a 1D DMA transfer (non-strided)
def _num_datatrcmds_fetch_tile_data(layer, params_exec, dma_type='1D'):
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    
    if layer['optype'] in (OPTYPES.O_CONV1D_DW, OPTYPES.O_CONV2D_DW):
        # block size for each DMA transfer per buffer type
        # Always use Tm as Tn=1
        blkI = Tm
        blkW = 1  # Always fetch one channel only, as each channel is fetched separately
        blkO = Tm

        # num of transfers per buffer type
        nI = Tri * Tci
        nW = Kh * Kw * Tm
        nO = Tr * Tc
    elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
        
        # block size for each DMA transfer per buffer type
        blkI = Tn
        blkW = Tn
        blkO = Tm

        # num of transfers per buffer type
        nI = Tri * Tci
        nW = Kh * Kw * Tm
        nO = Tr * Tc

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
# MAIN COST MODEL - CONTINUOUS POWER
############################################################################

def est_cost_CONV_contpow(layer, params_exec, plat_settings, plat_cost_profile):  
    #E_rb, L_rb = est_cost_CONV_reboot_contpow(plat_cost_profile) # not included in model
    E_fd, L_fd = est_cost_CONV_layerinputfetch_contpow(layer, params_exec, plat_cost_profile)    
    E_cp, L_cp = est_cost_CONV_layercomp_contpow(layer, params_exec, plat_cost_profile)
    #E_bd, L_bd = est_cost_CONV_layeroutputbackup_contpow(layer, params_exec, plat_cost_profile) 
    # no tile indeces fetching/preserving in this model (cont pow)
    
    cost_breakdown={
        "rb": [0, 0],
        "fd": [E_fd, L_fd],  #forword
        "fl": [0, 0],
        "cp": [E_cp, L_cp],  #compute
        "bd": [0, 0],  #backup
        "bl": [0, 0]
    }

    # only considering, [fetch -> compute -> save result] per layer
    total_energy = E_fd + E_cp #+ E_bd
    total_latency = L_fd + L_cp #+ L_bd

    return total_energy, total_latency, cost_breakdown

def est_cost_CONV_reboot_contpow(layer, plat_cost_profile):    
    total_en_cost, total_lat_cost = est_cost_CONV_reboot_intpow(layer, plat_cost_profile) # same as intermittent power
    return total_en_cost, total_lat_cost


# get layer data input fetch energy cost (overall layer)
def est_cost_CONV_layerinputfetch_contpow(layer, params_exec, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0  
       
    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
       
    # num of data transfer command invocations
    nI, nW, nO, blkI, blkW, blkO = _num_datatrcmds_fetch_tile_data(layer, params_exec)

    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, op_type=layer['optype'])
    Rtr = np.ceil(R/Tr); Ctc = np.ceil(C/Tc); Mtm = np.ceil(M/Tm); Ntn = np.ceil(N/Tn)

    ngt0 = (num_tiles-(Rtr*Ctc*Mtm)) # num iterations where n>0
        
    # energy/latency cost of the transfer (for a given block size)
    er_blkI = plat_cost_profile['E_DMA_NVM_TO_VM'](blkI); lr_blkI = plat_cost_profile['L_DMA_NVM_TO_VM'](blkI)
    er_blkW = plat_cost_profile['E_DMA_NVM_TO_VM'](blkW); lr_blkW = plat_cost_profile['L_DMA_NVM_TO_VM'](blkW)
    er_blkO = plat_cost_profile['E_DMA_NVM_TO_VM'](blkO); lr_blkO = plat_cost_profile['L_DMA_NVM_TO_VM'](blkO)
    
    # energy/latency overhead of each transfer
    eofI = plat_cost_profile['E_FD_I_OVHD']; lofI = plat_cost_profile['L_FD_I_OVHD']
    eofW = plat_cost_profile['E_FD_W_OVHD']; lofW = plat_cost_profile['L_FD_W_OVHD']
    
    eofO_ruI = plat_cost_profile['E_FD_O_RUI_OVHD']; lofO_ruI = plat_cost_profile['L_FD_O_RUI_OVHD']
    eofO_ruW = plat_cost_profile['E_FD_O_RUW_OVHD']; lofO_ruW = plat_cost_profile['L_FD_O_RUW_OVHD']
    eofO_ruO = plat_cost_profile['E_FD_O_RUO_OVHD']; lofO_ruO = plat_cost_profile['L_FD_O_RUO_OVHD']
    
    # XXX: 2023/12/01 cost model checked up to here

    # -- calc energy cost depending on reuse scheme        
    if layer['optype'] in (OPTYPES.O_CONV1D_DW, OPTYPES.O_CONV2D_DW):
        # ====================== Depthwise conv ==============================
        if inter_lo == "reuse_I":
            # DW conv does not need fetch and accumulate each channel, so no OFM fetch costs
            total_en_cost = ((Rtr*Ctc*Ntn)*(nI*(er_blkI + eofI))) + ((num_tiles)*(nW * (er_blkW + eofW))) + (0*(nO * (er_blkO + eofO_ruI)))
            total_lat_cost = ((Rtr*Ctc*Ntn)*(nI*(lr_blkI + lofI))) + ((num_tiles)*(nW * (lr_blkW + lofW))) + (0*(nO * (lr_blkO + lofO_ruI)))

        elif inter_lo == "reuse_W":
            total_en_cost = ((num_tiles)*(nI*(er_blkI + eofI))) + ((Mtm)*(nW * (er_blkW + eofW))) + (0*(nO * (er_blkO + eofO_ruW)))
            total_lat_cost = ((num_tiles)*(nI*(lr_blkI + lofI))) + ((Mtm)*(nW * (lr_blkW + lofW))) + (0*(nO * (lr_blkO + lofO_ruW)))
                            
        # partial sums are always in VM, so no need to refetch
        elif inter_lo == "reuse_O":
            total_en_cost = ((num_tiles)*(nI*(er_blkI + eofI))) + ((num_tiles)*(nW * (er_blkW + eofW)))
            total_lat_cost = ((num_tiles)*(nI*(lr_blkI + lofI))) + ((num_tiles)*(nW * (lr_blkW + lofW)))
                    
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")

    elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
        # ====================== Standard conv ==============================
        if inter_lo == "reuse_I":
            total_en_cost = ((Rtr*Ctc*Ntn)*(nI*(er_blkI + eofI))) + ((num_tiles)*(nW * (er_blkW + eofW))) + (ngt0*(nO * (er_blkO + eofO_ruI)))
            total_lat_cost = ((Rtr*Ctc*Ntn)*(nI*(lr_blkI + lofI))) + ((num_tiles)*(nW * (lr_blkW + lofW))) + (ngt0*(nO * (lr_blkO + lofO_ruI)))

        elif inter_lo == "reuse_W":
            total_en_cost = ((num_tiles)*(nI*(er_blkI + eofI))) + ((Mtm*Ntn)*(nW * (er_blkW + eofW))) + (ngt0*(nO * (er_blkO + eofO_ruW)))
            total_lat_cost = ((num_tiles)*(nI*(lr_blkI + lofI))) + ((Mtm*Ntn)*(nW * (lr_blkW + lofW))) + (ngt0*(nO * (lr_blkO + lofO_ruW)))
                            
        # partial sums are always in VM, so no need to refetch
        elif inter_lo == "reuse_O":
            total_en_cost = ((num_tiles)*(nI*(er_blkI + eofI))) + ((num_tiles)*(nW * (er_blkW + eofW)))
            total_lat_cost = ((num_tiles)*(nI*(lr_blkI + lofI))) + ((num_tiles)*(nW * (lr_blkW + lofW)))
                    
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown op_type: " + OPTYPES.get_optype_label(layer['optype']))

    return total_en_cost, total_lat_cost


# backup layer output energy cost (overall layer)
def est_cost_CONV_layeroutputbackup_contpow(layer, params_exec, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0  
       
    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
       
    # num of data transfer command invocations
    nO, blkO = _num_datatrcmds_backup_tile_data(layer, params_exec)

    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, op_type=layer['optype'])
    Rtr = np.ceil(R/Tr); Ctc = np.ceil(C/Tc); Mtm = np.ceil(M/Tm); Ntn = np.ceil(N/Tn)

    params_pres = {'backup_batch_size': 1} # S=1
    tile_en_cost, tile_lat_cost = est_cost_CONV_tileoutputbackup_intpow(layer, params_exec, params_pres, plat_cost_profile) # cost for one tile # same as intermittent power, but S=1

    total_en_cost = tile_en_cost * num_tiles
    total_lat_cost = tile_lat_cost * num_tiles

    return total_en_cost, total_lat_cost


def est_cost_CONV_layercomp_contpow(layer, params_exec, plat_cost_profile):    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    H, W, R, C, M, N, Kh, Kw, stride = common._get_layer_props(layer)
    num_tiles = common._num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, op_type=layer['optype'])
    
    params_pres = {'backup_batch_size': 1} # S=1
    tile_en_cost, tile_lat_cost = est_cost_CONV_tilecomp_intpow(layer, params_exec, params_pres, plat_cost_profile) # cost for one tile # same as intermittent power, but S=1
    
    total_en_cost = tile_en_cost * num_tiles
    total_lat_cost = tile_lat_cost * num_tiles

    return total_en_cost, total_lat_cost


############################################################################
# MAIN COST MODEL - INTERMITTENT POWER (per power cycle)
############################################################################
# get estimated total energy consumption in a power cycle
def est_cost_CONV_powcycle_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, exclude_rb_cost=False, return_only_breakdown=False):        
    E_rb, L_rb = est_cost_CONV_reboot_intpow(layer, plat_cost_profile)
    E_fd_ngt0, L_fd_ngt0, E_fd_n0, L_fd_n0 = est_cost_CONV_tileinputfetch_intpow(layer, params_exec, params_pres, plat_cost_profile)
    E_fl, L_fl = est_cost_CONV_tileidxfetch_intpow(layer, plat_cost_profile)
    E_cp, L_cp = est_cost_CONV_tilecomp_intpow(layer, params_exec, params_pres, plat_cost_profile)
    E_bd, L_bd = est_cost_CONV_tileoutputbackup_intpow(layer, params_exec, params_pres, plat_cost_profile)
    E_bl, L_bl = est_cost_CONV_tileidxbackup_intpow(layer, plat_cost_profile)

    total_energy_powcycle_ngt0 = E_rb + E_fd_ngt0 + E_fl + E_cp + E_bd + E_bl
    total_latency_powcycle_ngt0 = L_rb + L_fd_ngt0 + L_fl + L_cp + L_bd + L_bl

    total_energy_powcycle_n0 = E_rb + E_fd_n0 + E_fl + E_cp + E_bd + E_bl
    total_latency_powcycle_n0 = L_rb + L_fd_n0 + L_fl + L_cp + L_bd + L_bl
    
    cost_breakdown={
        "rb": [E_rb, L_rb],
        #"fd": [E_fd_ngt0+E_fd_n0, L_fd_ngt0+L_fd_n0],
        "fd": [E_fd_ngt0, L_fd_ngt0],   # assume MAX cost
        "fl": [E_fl, L_fl],
        "cp": [E_cp, L_cp],
        "bd": [E_bd, L_bd],
        "bl": [E_bl, L_bl]
    }

    if (return_only_breakdown == False):
        return total_energy_powcycle_ngt0, total_latency_powcycle_ngt0, total_energy_powcycle_n0, total_latency_powcycle_n0, cost_breakdown
    else:        
        return E_rb, L_rb, E_fd_ngt0, L_fd_ngt0, E_fd_n0, L_fd_n0, E_fl, L_fl, E_cp, L_cp, E_bd, L_bd, E_bl, L_bl
    

# get reboot energy cost
def est_cost_CONV_reboot_intpow(layer, plat_cost_profile):
    total_en_cost = plat_cost_profile['E_RB']
    total_lat_cost = plat_cost_profile['L_RB']
    return total_en_cost, total_lat_cost


# get tile data input fetch energy cost per power cycle
def est_cost_CONV_tileinputfetch_intpow(layer, params_exec, params_pres, plat_cost_profile):
    total_en_cost_ngt0 = 0    
    total_lat_cost_ngt0 = 0  
    total_en_cost_n0 = 0    
    total_lat_cost_n0 = 0    

    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # preservation space   
    S = params_pres['backup_batch_size']
   
    # num of data transfer command invocations
    nI, nW, nO, blkI, blkW, blkO = _num_datatrcmds_fetch_tile_data(layer, params_exec)
    #print(nI, nW, nO, blkI, blkW, blkO)
    
    # energy/latency cost of the transfer (for a given block size)
    #er_Tn = plat_cost_profile['E_DMA_NVM_TO_VM'](Tn); lr_Tn = plat_cost_profile['L_DMA_NVM_TO_VM'](Tn)
    #er_Tm = plat_cost_profile['E_DMA_NVM_TO_VM'](Tm); lr_Tm = plat_cost_profile['L_DMA_NVM_TO_VM'](Tm)
    
    er_blkI = plat_cost_profile['E_DMA_NVM_TO_VM'](blkI); lr_blkI = plat_cost_profile['L_DMA_NVM_TO_VM'](blkI)
    er_blkW = plat_cost_profile['E_DMA_NVM_TO_VM'](blkW); lr_blkW = plat_cost_profile['L_DMA_NVM_TO_VM'](blkW)
    er_blkO = plat_cost_profile['E_DMA_NVM_TO_VM'](blkO); lr_blkO = plat_cost_profile['L_DMA_NVM_TO_VM'](blkO)
    
    # energy/latency overhead of each transfer
    eofI = plat_cost_profile['E_FD_I_OVHD']; lofI = plat_cost_profile['L_FD_I_OVHD']
    eofW = plat_cost_profile['E_FD_W_OVHD']; lofW = plat_cost_profile['L_FD_W_OVHD']    
    eofO_ruI = plat_cost_profile['E_FD_O_RUI_OVHD']; lofO_ruI = plat_cost_profile['L_FD_O_RUI_OVHD']
    eofO_ruW = plat_cost_profile['E_FD_O_RUW_OVHD']; lofO_ruW = plat_cost_profile['L_FD_O_RUW_OVHD']
    eofO_ruO = plat_cost_profile['E_FD_O_RUO_OVHD']; lofO_ruO = plat_cost_profile['L_FD_O_RUO_OVHD']    

    # -- calc energy cost depending on reuse scheme        
    if layer['optype'] in (OPTYPES.O_CONV1D_DW, OPTYPES.O_CONV2D_DW):
        # =========================== Depthwise conv ==========================
        if inter_lo == "reuse_I":
            # Special case: IFM reuse is not helpful for DW conv - see the inference engine codes
            total_en_cost_ngt0 = (S * ((nW * (er_blkW + eofW)) )) + S*(nI*(er_blkI + eofI))
            total_lat_cost_ngt0 = (S * ((nW * (lr_blkW + lofW)) )) + S*(nI*(lr_blkI + lofI))
            total_en_cost_n0 = (S * ((nW * (er_blkW + eofW)))) + S*(nI*(er_blkI + eofI))
            total_lat_cost_n0 = (S * ((nW * (lr_blkW + lofW)))) + S*(nI*(lr_blkI + lofI))

        elif inter_lo in ("reuse_W", "reuse_O"):
            # For Depthwise
            total_en_cost_ngt0 = (S * ((nI * (er_blkI + eofI)) )) + (nW*(er_blkW + eofW))
            total_lat_cost_ngt0 = (S * ((nI * (lr_blkI + lofI)) )) + (nW*(lr_blkW + lofW))
            total_en_cost_n0 = (S * (nI * (er_blkI + eofI)) ) + (nW*(er_blkW + eofW))
            total_lat_cost_n0 = (S * (nI * (lr_blkI + lofI)) ) + (nW*(lr_blkW + lofW))
            
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")
    elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
        # =========================== Standard conv ==========================
        if inter_lo == "reuse_I":
            total_en_cost_ngt0 = (S * ((nW * (er_blkW + eofW)) + (nO * (er_blkO + eofO_ruI)))) + (nI*(er_blkI + eofI))
            total_lat_cost_ngt0 = (S * ((nW * (lr_blkW + lofW)) + (nO * (lr_blkO + lofO_ruI)))) + (nI*(lr_blkI + lofI))
            total_en_cost_n0 = (S * ((nW * (er_blkW + eofW)))) + (nI*(er_blkI + eofI))
            total_lat_cost_n0 = (S * ((nW * (lr_blkW + lofW)))) + (nI*(lr_blkI + lofI))

        elif inter_lo == "reuse_W":
            total_en_cost_ngt0 = (S * ((nI * (er_blkI + eofI)) + (nO * (er_blkO + eofO_ruW)))) + (nW*(er_blkW + eofW))
            total_lat_cost_ngt0 = (S * ((nI * (lr_blkI + lofI)) + (nO * (lr_blkO + lofO_ruW)))) + (nW*(lr_blkW + lofW))
            total_en_cost_n0 = (S * (nI * (er_blkI + eofI)) ) + (nW*(er_blkW + eofW))
            total_lat_cost_n0 = (S * (nI * (lr_blkI + lofI)) ) + (nW*(lr_blkW + lofW))
            
        elif inter_lo == "reuse_O":
            total_en_cost_ngt0 = (S * ((nI * (er_blkI + eofI)) + (nW * (er_blkW + eofW)))) + (nO*(er_blkO + eofO_ruO))
            total_lat_cost_ngt0 = (S * ((nI * (lr_blkI + lofI)) + (nW * (lr_blkW + lofW)))) + (nO*(lr_blkO + lofO_ruO))
            total_en_cost_n0 = (S * ((nI * (er_blkI + eofI)) + (nW * (er_blkW + eofW))))
            total_lat_cost_n0 = (S * ((nI * (lr_blkI + lofI)) + (nW * (lr_blkW + lofW))))
            
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown op_type: " + OPTYPES.get_optype_label(layer['optype']))

    #print(inter_lo, total_en_cost_ngt0)
    #input()
    
    return total_en_cost_ngt0, total_lat_cost_ngt0, total_en_cost_n0, total_lat_cost_n0


# get tile data output backup energy cost per power cycle
def est_cost_CONV_tileoutputbackup_intpow(layer, params_exec, params_pres, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0    

    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # preservation space   
    S = params_pres['backup_batch_size']
   
    # num of data transfer command invocations
    #nO, blkO = _num_datatrcmds_backup_tile_data(params_exec)
    # energy/latency cost of the transfer (for a given block size)    
    ew_Tm = plat_cost_profile['E_DMA_VM_TO_NVM'](Tm); lw_Tm = plat_cost_profile['L_DMA_VM_TO_NVM'](Tm)
    ew_STm = plat_cost_profile['E_DMA_VM_TO_NVM'](S*Tm); lw_STm = plat_cost_profile['L_DMA_VM_TO_NVM'](S*Tm)
    # energy/latency overhead of each transfer    
    eobO_ruI = plat_cost_profile['E_BD_O_RUI_OVHD']; lobO_ruI = plat_cost_profile['L_BD_O_RUI_OVHD']
    eobO_ruW = plat_cost_profile['E_BD_O_RUW_OVHD']; lobO_ruW = plat_cost_profile['L_BD_O_RUW_OVHD']
    eobO_ruO = plat_cost_profile['E_BD_O_RUO_OVHD']; lobO_ruO = plat_cost_profile['L_BD_O_RUO_OVHD']

    # -- calc energy cost depending on reuse scheme        
    # Same for both standard and DW conv
    if (inter_lo == "reuse_I"):
        total_en_cost = Tr * Tc * (ew_STm + eobO_ruI) 
        total_lat_cost = Tr * Tc * (lw_STm + lobO_ruI)
    elif (inter_lo == "reuse_W"):
        total_en_cost = S * Tr * Tc * (ew_Tm + eobO_ruW) 
        total_lat_cost = S * Tr * Tc * (lw_Tm + lobO_ruW) 
    elif inter_lo == "reuse_O":
        total_en_cost = Tr * Tc * (ew_Tm + eobO_ruO) 
        total_lat_cost = Tr * Tc * (lw_Tm + lobO_ruO) 
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")
    
    return total_en_cost, total_lat_cost


def est_cost_CONV_tilecomp_intpow(layer, params_exec, params_pres, plat_cost_profile):
    total_en_cost = 0    
    total_lat_cost = 0

    # execution space
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']    
    # preservation space   
    S = params_pres['backup_batch_size']
    
    eadd = plat_cost_profile['E_ADD']
    emul = plat_cost_profile['E_MUL']
    ladd = plat_cost_profile['L_ADD']
    lmul = plat_cost_profile['L_MUL']

    if layer['optype'] in (OPTYPES.O_CONV2D_DW, OPTYPES.O_CONV1D_DW):
        # ===================== Depthwise conv ===================
        ecomp_ovh_other_reuse = plat_cost_profile['E_OP_COMP_DWCONV_OVHD_OTHER'] # addressing overhead
        ecomp_ovh_ifm_reuse = plat_cost_profile['E_OP_COMP_DWCONV_OVHD_IFM'] # addressing overhead

        lcomp_ovh_other_reuse = plat_cost_profile['L_OP_COMP_DWCONV_OVHD_OTHER'] # addressing overhead
        lcomp_ovh_ifm_reuse = plat_cost_profile['L_OP_COMP_DWCONV_OVHD_IFM'] # addressing overhead

        if inter_lo == "reuse_I":
            ecomp_ovh = ecomp_ovh_ifm_reuse
            lcomp_ovh = lcomp_ovh_ifm_reuse
        else:
            ecomp_ovh = ecomp_ovh_other_reuse
            lcomp_ovh = lcomp_ovh_other_reuse
        
        total_en_cost = S * Tr * Tc * Tm * Kh * Kw * (emul + ecomp_ovh)
        total_lat_cost = S * Tr * Tc * Tm * Kh * Kw * (lmul + lcomp_ovh)        
            
    elif layer['optype'] in (OPTYPES.O_CONV1D, OPTYPES.O_CONV2D, OPTYPES.O_CONV1D_PW, OPTYPES.O_CONV2D_PW, OPTYPES.O_FC):
        # ===================== Standard conv ===================
        evmac = plat_cost_profile['E_OP_VECMAC'](Tn)
        ecomp_ovh = plat_cost_profile['E_OP_COMP_CONV_OVHD'] # addressing overhead

        lvmac = plat_cost_profile['L_OP_VECMAC'](Tn)
        lcomp_ovh = plat_cost_profile['L_OP_COMP_CONV_OVHD'] # addressing overhead        
        
        total_en_cost = S * (Kh * Kw * Tr * Tc * Tm * (evmac + eadd + ecomp_ovh) + 5*emul)
        total_lat_cost = S * (Kh * Kw * Tr * Tc * Tm * (lvmac + ladd + lcomp_ovh) + 5*lmul)

    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown op_type")

    return total_en_cost, total_lat_cost

# cost of fetch inter tile indices
def est_cost_CONV_tileidxfetch_intpow(layer, plat_cost_profile):
    total_en_cost = plat_cost_profile['E_DMA_NVM_TO_VM'](5)    #[4 tile indeces + 1 layer index]
    total_lat_cost = plat_cost_profile['L_DMA_NVM_TO_VM'](5)        
    return total_en_cost, total_lat_cost


# cost of backup inter tile indices
def est_cost_CONV_tileidxbackup_intpow(layer, plat_cost_profile):
    total_en_cost = plat_cost_profile['E_DMA_VM_TO_NVM'](5)   #[4 tile indeces + 1 layer index]
    total_lat_cost = plat_cost_profile['L_DMA_VM_TO_NVM'](5)        
    return total_en_cost, total_lat_cost


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
    









    

















