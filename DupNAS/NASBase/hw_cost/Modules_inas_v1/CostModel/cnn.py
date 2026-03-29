import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect
import itertools
import operator


# local imports
from . import common
from ....model.common_types import OPTYPES
from .conv_tiled import est_cost_CONV_powcycle_intpow, est_cost_CONV_contpow, est_cost_CONV_reboot_contpow, est_cost_CONV_flops
from .pool_tiled import est_cost_POOL_powcycle_intpow, est_cost_POOL_contpow, est_cost_GAVGPOOL_flops
from .bn_tiled import est_cost_BN_powcycle_intpow, est_cost_BN_contpow, est_cost_BN_flops
from .relu_tiled import est_cost_RELU_powcycle_intpow, est_cost_RELU_contpow
from .add_tiled import est_cost_ADD_powcycle_intpow, est_cost_ADD_contpow, est_cost_ADD_flops
from .capacitor import cal_cap_recharge_time_custom, cap_energy


DEBUG_CONSTRAINTS = False


############################################################################
# HELPERS
############################################################################

def report_nvm_constraints(network_nvm_usage, plat_settings):
    if not DEBUG_CONSTRAINTS:
        return

    max_features_req, total_weights_req = network_nvm_usage[-1]
    del network_nvm_usage[-1]

    network_nvm_usage = [(layer_idx, nvm_features_req, nvm_weights_req)
                         for layer_idx, (nvm_features_req, nvm_weights_req)
                         in zip(range(len(network_nvm_usage)), network_nvm_usage)]

    nvm_capacity = plat_settings['NVM_CAPACITY']
    nvm_capacity_allocation = plat_settings['NVM_CAPACITY_ALLOCATION']
    features_capacity, weights_capacity = nvm_capacity_allocation

    def report_features(limit=None):
        network_nvm_usage.sort(key=operator.itemgetter(1), reverse=True)  # Sort by item 1 (nvm_features_req)
        for layer_idx, nvm_features_req, nvm_weights_req in itertools.islice(network_nvm_usage, limit):
            if limit is None and nvm_features_req < features_capacity:
                break
            #print(f"Layer {layer_idx}, nvm_features_req {nvm_features_req}")

    def report_weights(limit=None):
        network_nvm_usage.sort(key=operator.itemgetter(2), reverse=True)  # Sort by item 2 (nvm_weights_req)
        accumulated_weights_req = 0
        for layer_idx, nvm_features_req, nvm_weights_req in itertools.islice(network_nvm_usage, limit):
            #print(f"Layer {layer_idx}, nvm_weights_req {nvm_weights_req}")
            accumulated_weights_req += nvm_weights_req
            if accumulated_weights_req > weights_capacity:
                break

    if max_features_req > features_capacity:
        print(f"max_features_req {max_features_req} exceeds NVM capacity for features {features_capacity}")
        report_features()

    if total_weights_req > weights_capacity:
        print(f"total_weights_req {total_weights_req} exceeds NVM capacity for weights {weights_capacity}")
        report_weights()

    if max_features_req <= features_capacity and total_weights_req <= weights_capacity and max_features_req + total_weights_req > nvm_capacity:
        print(f"max_features_req {max_features_req} + total_weights_req {total_weights_req} exceeds NVM capacity {nvm_capacity}")


############################################################################
# CONSTRAINT CHECKING
############################################################################
# will the min exec and pres params make forward progress under given energy budget ?
def pass_constraint_atomicity(Epc, plat_settings):
    Eav = cap_energy(plat_settings['CCAP'], plat_settings['VON'], plat_settings['VOFF']) * plat_settings['EAV_SAFE_MARGIN']
    if Epc > Eav:
        return [False, Eav, Epc]
    else:
        return [True, Eav, Epc]

# will the min tile size fit into the available volatile memory of the system ?
def pass_constraint_spatial(layer, plat_settings, params_exec, params_pres):
    #vm_capacity = plat_settings['VM_CAPACITY']
    vm_capacity = plat_settings['VM_CONSTRAINT']
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']
    inter_lo = params_exec['inter_lo']
    S = params_pres['backup_batch_size']
    
    B_in, B_w, B_out = common._vm_buff_size(Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn, inter_lo, S, layer_type = layer['type'], op_type=layer['optype'])
    total_vm_req = (B_in + B_w + B_out) * plat_settings['DATA_SZ']
    #print("total_vm_req = ",total_vm_req, " B_in , B_w , B_out = ", B_in , B_w , B_out, " plat_settings['DATA_SZ'] = ", plat_settings['DATA_SZ'])
    if total_vm_req > vm_capacity:
        print("False: total_vm_req > vm_capacity", total_vm_req, vm_capacity)
        return [False, vm_capacity, total_vm_req]
    else:
        return [True, vm_capacity, total_vm_req]


# NVM buffer requirements during inference
## copy this func, -> def get_spatial_nvm_no_write_back
def get_spatial_nvm(layer, plat_settings):
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    H = layer['IFM'].h; W = layer['IFM'].w
    Kw = layer['K'].w;  Kh = layer['K'].h

    # Use conservative memory allocation (there might be some wasted space)
    # For each layer, two buffers are needed for double buffering (one for input and one for output).
    # ADD needs one more input buffer as both input feature maps are on NVM.
    # This allocation strategy works for MobileNetV2, as there is at most one shortcut (skip enabled).
    ofm_size = M * R * C
    ifm_size = N * H * W
    assert (ofm_size != 0 and ifm_size != 0)

    # Calculating IFM & OFM sizes
    if layer['lcnt']:
        lcnt_value = layer['lcnt'].split('/')[0]
        if lcnt_value == 0:
            nvm_features_req = ifm_size
        else:
            nvm_features_req = 0
    # if layer['type'] in ('CONV', 'FC', 'POOL', 'GAVGPOOL', 'BN', 'RELU'):
    #     # BN, RELU may be implemented with in-place update, while the NVM consumption is still under 2 * max(ifm_size, ofm_size)
    #     nvm_features_req = 2 * max(ifm_size, ofm_size) 
    # elif layer['type'] in ('ADD',):
    #     nvm_features_req = 3 * max(ifm_size, ofm_size)
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer value")

    # Calculating weight size
    if layer['type'] in ('CONV', 'FC'):
        if layer['optype'] in (OPTYPES.O_CONV2D_DW, OPTYPES.O_CONV1D_DW):
            nvm_weights_req = M * Kh * Kw
        else:
            nvm_weights_req = M * N * Kh * Kw
        assert nvm_weights_req != 0
    elif layer['type'] in ('POOL', 'GAVGPOOL', 'RELU', 'ADD'):
        nvm_weights_req = 0
    elif layer['type'] in ('BN',):
        nvm_weights_req = M * 4  # mu, sigma, beta (weight), gamma (bias)
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")

    # *2 for Q15
    nvm_features_req *= plat_settings['DATA_SZ']
    nvm_weights_req *= plat_settings['DATA_SZ']

    return [nvm_features_req, nvm_weights_req]

def pass_constraint_storage(network, plat_settings):
    # Check if NVM is enough
    nvm_capacity = plat_settings['NVM_CAPACITY']
    nvm_capacity_allocation = plat_settings['NVM_CAPACITY_ALLOCATION']
    features_capacity, weights_capacity = nvm_capacity_allocation

    network_nvm_usage = []
    max_features_req = 0
    total_weights_req = 0

    # exec and pres params not given, so have to find best sol
    for lidx, each_layer in enumerate(network):
        layer_nvm_usage = get_spatial_nvm(each_layer, plat_settings)

        nvm_features_req, nvm_weights_req = layer_nvm_usage
        if (nvm_features_req > features_capacity):
            # for didx, du in enumerate(dup_path):
            #     #print("layer: ", lidx, " nvm_features_req: ", nvm_features_req)
            #     if lidx>=du['start'] and lidx<=du['end']:
            #         #print("layer: ", lidx, "in dup path ", du, " max est_peak_per_path = ", est_peak_per_path[didx])
            #         nvm_features_req = est_peak_per_path[didx]
            break

        max_features_req = max(max_features_req, nvm_features_req)
        total_weights_req += nvm_weights_req

        network_nvm_usage.append(layer_nvm_usage)

    network_nvm_usage.append([max_features_req, total_weights_req])

    # two FRAM chips model, one for features and another for weights
    all_layers_fit_nvm = (max_features_req <= features_capacity) and (total_weights_req <= weights_capacity) and (max_features_req + total_weights_req <= nvm_capacity)
    #print("mfr, mwr, total = ", max_features_req, total_weights_req, max_features_req + total_weights_req)
    if not all_layers_fit_nvm:
        report_nvm_constraints(network_nvm_usage, plat_settings)

    return [all_layers_fit_nvm, nvm_capacity_allocation, network_nvm_usage]

# is the min achievable E2E latency lower than the latency constraint ?
def pass_constraint_responsiveness(L_e2e, plat_settings):
    lat_e2e_req = plat_settings['LAT_E2E_REQ']
    if L_e2e == -1 or L_e2e > lat_e2e_req:
        if DEBUG_CONSTRAINTS:
            if L_e2e != -1:
                print(f"Network has latency {L_e2e}, which exceeds latency constraint {lat_e2e_req}")
        return [False, lat_e2e_req, L_e2e]
    else:
        return [True, lat_e2e_req, L_e2e]

def pass_constraint_imc(imc, plat_settings):
    imc_constraint = plat_settings['IMC_CONSTRAINT']
    if imc == -1 or imc > imc_constraint:
        if DEBUG_CONSTRAINTS:
            if imc != -1:
                print(f"Network has IMC {imc}, which exceeds IMC constraint {imc_constraint}")
        return [False, imc_constraint, imc]
    else:
        return [True, imc_constraint, imc]


############################################################################
# COST ANALYSIS (continuous power)
############################################################################
# --- SINGLE LAYER ----
def est_cost_layer_contpow(layer, params_exec, plat_settings, plat_cost_profile):    
    layer_type = layer['type']        
    R = layer['OFM'].h; C = layer['OFM'].h; M = layer['OFM'].ch; N = layer['IFM'].ch    
    inter_lo = params_exec['inter_lo']    
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']
        
    # get per layer energy and latency
    if layer_type == "CONV":            
        lay_E, lay_L, cost_breakdown = est_cost_CONV_contpow(layer, params_exec, plat_settings, plat_cost_profile)                        

    elif (layer_type == "POOL") or (layer_type == "GAVGPOOL"):        
        lay_E, lay_L, cost_breakdown = est_cost_POOL_contpow(layer, params_exec, plat_settings, plat_cost_profile)                      

    elif layer_type == "FC":        
        lay_E, lay_L, cost_breakdown = est_cost_CONV_contpow(layer, params_exec, plat_settings, plat_cost_profile)          
            
    elif layer_type == "BN":        
        lay_E, lay_L, cost_breakdown = est_cost_BN_contpow(layer, params_exec, plat_settings, plat_cost_profile)  
            
    elif layer_type == "RELU":        
        lay_E, lay_L, cost_breakdown = est_cost_RELU_contpow(layer, params_exec, plat_settings, plat_cost_profile)          
        
    elif layer_type == "ADD":        
        lay_E, lay_L, cost_breakdown = est_cost_ADD_contpow(layer, params_exec, plat_settings, plat_cost_profile)                  

    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
    
    # # get per layer energy and latency
    # if layer_type == "CONV":        
    #     if (common.check_conv(layer)):
    #         lay_E, lay_L = est_cost_CONV_contpow(layer, params_exec, plat_settings, plat_cost_profile)                
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - CONV dimensions incorrect") 

    # elif (layer_type == "POOL") or (layer_type == "GAVGPOOL"):
    #     if (common.check_pool(layer)):
    #         lay_E, lay_L = est_cost_POOL_contpow(layer, params_exec, plat_settings, plat_cost_profile)              
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - POOL dimensions incorrect") 

    # elif layer_type == "FC":
    #     if (common.check_fc(layer)):
    #         lay_E, lay_L = est_cost_CONV_contpow(layer, params_exec, plat_settings, plat_cost_profile)  
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - FC dimensions incorrect") 
            
    # elif layer_type == "BN":
    #     if (common.check_bn(layer)):
    #         lay_E, lay_L = est_cost_BN_contpow(layer, params_exec, plat_settings, plat_cost_profile)  
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - BN dimensions incorrect") 
    
    # elif layer_type == "RELU":
    #     if (common.check_relu(layer)):
    #         lay_E, lay_L = est_cost_RELU_contpow(layer, params_exec, plat_settings, plat_cost_profile)  
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - RELU dimensions incorrect") 

    # else:
    #     sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
        
    return lay_E, lay_L, cost_breakdown

def est_cost_onetime_reboot(plat_cost_profile):
    en_cost, lat_cost = est_cost_CONV_reboot_contpow(None, plat_cost_profile)
    return en_cost, lat_cost



############################################################################
# COST ANALYSIS (intermittent power)
############################################################################
# --- SINGLE LAYER ----
def est_cost_layer_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile):     
    layer_type = layer['type']            
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch    
    H = layer['IFM'].h; W = layer['IFM'].w
    inter_lo = params_exec['inter_lo']
    S = params_pres['backup_batch_size']
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']
    npc, npc_n0, npc_ngt0 = common._num_pow_cycles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, S, inter_lo, layer_type = layer['type'], op_type=layer['optype'])
    
    # get per power cycle energy and latency        
    if layer_type == "CONV":        
        Epc_max, Lpc_max, Epc_min, Lpc_min, cost_breakdown = est_cost_CONV_powcycle_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile)                
        
    elif (layer_type == "POOL") or (layer_type == "GAVGPOOL"):        
        Epc_min, Lpc_min, cost_breakdown = est_cost_POOL_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
        Epc_max=Epc_min; Lpc_max=Lpc_min
        
    elif layer_type == "FC":        
        Epc_max, Lpc_max, Epc_min, Lpc_min, cost_breakdown = est_cost_CONV_powcycle_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile)  
        
    elif layer_type == "BN":        
        Epc_min, Lpc_min, cost_breakdown = est_cost_BN_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
                    
    elif layer_type == "RELU":
        Epc_min, Lpc_min, cost_breakdown = est_cost_RELU_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
    
    elif layer_type == "ADD":
        Epc_min, Lpc_min, cost_breakdown = est_cost_ADD_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
    
    # if no power cycle contains n>0 iterations, then Epc is same in every power cycle
    if npc_ngt0 == 0:
        Epc_max = Epc_min
        Lpc_max = Lpc_min
        
    return Epc_max, Lpc_max, Epc_min, Lpc_min, cost_breakdown


    # # get per power cycle energy and latency        
    # if layer_type == "CONV":        
    #     if (common.check_conv(layer)):
    #         Epc_max, Lpc_max, Epc_min, Lpc_min = est_cost_CONV_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)                
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - CONV dimensions incorrect") 

    # elif (layer_type == "POOL") or (layer_type == "GAVGPOOL"):
    #     if (common.check_pool(layer)):
    #         Epc_min, Lpc_min = est_cost_POOL_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
    #         Epc_max=Epc_min; Lpc_max=Lpc_min
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - POOL dimensions incorrect") 

    # elif layer_type == "FC":
    #     if (common.check_fc(layer)):
    #         Epc_max, Lpc_max, Epc_min, Lpc_min = est_cost_CONV_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - FC dimensions incorrect") 

    # elif layer_type == "BN":
    #     if (common.check_bn(layer)):
    #         Epc_min, Lpc_min = est_cost_BN_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - BN dimensions incorrect") 
            
    # elif layer_type == "RELU":
    #     if (common.check_relu(layer)):
    #         Epc_min, Lpc_min = est_cost_RELU_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile)  
    #     else:
    #         sys.exit(inspect.currentframe().f_code.co_name+"::Error - RELU dimensions incorrect") 

    # else:
    #     sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")

    # # if no power cycle contains n>0 iterations, then Epc is same in every power cycle
    # if npc_ngt0 == 0:
    #     Epc_max = Epc_min
    #     Lpc_max = Lpc_min
        
    # return Epc_max, Lpc_max, Epc_min, Lpc_min

# estimate single layer E2E latency
def est_latency_e2e_layer_intpow(layer, Epc_max, Lpc_max, Epc_min, Lpc_min, plat_settings, params_exec, params_pres):
    H, W, R, C, M, N, Kh, Kw, stride =  common._get_layer_props(layer)
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']
    inter_lo = params_exec['inter_lo']
    S = params_pres['backup_batch_size']
    Ccap=plat_settings['CCAP']; Rehm=plat_settings['REHM']; Vsup=plat_settings['VSUP']; Von=plat_settings['VON']; Voff=plat_settings['VOFF']
    safe_margin=plat_settings['EAV_SAFE_MARGIN']
    
    recharge_time_min = cal_cap_recharge_time_custom(Epc_min, Ccap, Rehm, Vsup, Von, Voff, safe_margin)
    recharge_time_max = cal_cap_recharge_time_custom(Epc_max, Ccap, Rehm, Vsup, Von, Voff, safe_margin)

    npc, npc_n0, npc_ngt0 = common._num_pow_cycles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, S, inter_lo, layer_type = layer['type'], op_type=layer['optype'])

    # all power on (system active) + power off (recharge durations)
    # considering that some power cycles the energy consumption and recharge duration will be lower    
    tot_latency = (npc_n0 * (Lpc_min + recharge_time_min)) + (npc_ngt0 * (Lpc_max + recharge_time_max))

    return tot_latency, npc, npc_n0, npc_ngt0, recharge_time_min, recharge_time_max

def est_npc_layer_intpow(layer, params_exec, params_pres):
    H, W, R, C, M, N, Kh, Kw, stride =  common._get_layer_props(layer)
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']
    inter_lo = params_exec['inter_lo']
    S = params_pres['backup_batch_size']
    npc, npc_n0, npc_ngt0 = common._num_pow_cycles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, S, inter_lo, layer_type = layer['type'], op_type=layer['optype'])
    return npc, npc_n0, npc_ngt0

 # ---- NETWORK COST ----
def est_cost_alllayers_intpow(network, params_exec, params_pres, plat_settings, plat_cost_profile):
    alllayer_costs = []
    for each_layer in network:
        layer_type = each_layer['type']
        layer_name = each_layer['name']
        Epc_max, Lpc_max, Epc_min, Lpc_min, cost_breakdown = est_cost_layer_intpow(each_layer, params_exec, params_pres, plat_settings, plat_cost_profile)
        alllayer_costs.append([layer_name, layer_type, Epc_max, Lpc_max, Epc_min, Lpc_min])

    return alllayer_costs
        
def est_data_access_layer_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, all_pc=True):
    layer_type = layer['type']            
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch    
    H = layer['IFM'].h; W = layer['IFM'].w
    inter_lo = params_exec['inter_lo']
    S = params_pres['backup_batch_size']
    Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn = params_exec['tile_size']

    npc, npc_n0, npc_ngt0 = common._num_pow_cycles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, S, inter_lo, layer_type = layer['type'], op_type=layer['optype'])

    # get per power cycle energy and latency        
    if layer_type == "CONV":        
        if (common.check_conv(layer)):
            E_rb, L_rb, E_fd_ngt0, L_fd_ngt0, E_fd_n0, L_fd_n0, E_fl, L_fl, E_cp, L_cp, E_bd, L_bd, E_bl, L_bl = est_cost_CONV_powcycle_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, return_only_breakdown=True)                

            if (all_pc): # all power cycles
                total_nvm_read_cost_L = (L_fd_ngt0*npc_ngt0) + (L_fd_n0*npc_n0) + (L_fl*npc)
                total_nvm_write_cost_L = (L_bd + L_bl)*npc
                total_nvm_read_cost_E = (E_fd_ngt0*npc_ngt0) + (E_fd_n0*npc_n0) + (E_fl*npc)
                total_nvm_write_cost_E = (E_bd + E_bl)*npc          
            else:   # single power cycle
                total_nvm_read_cost_L = L_fd_ngt0 + L_fd_n0 + L_fl
                total_nvm_write_cost_L = L_bd + L_bl            
                total_nvm_read_cost_E = E_fd_ngt0 + E_fd_n0 + E_fl
                total_nvm_write_cost_E = E_bd + E_bl
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - CONV dimensions incorrect") 

    elif (layer_type == "POOL") or (layer_type == "GAVGPOOL"):
        if (common.check_pool(layer)):
            E_rb, L_rb, E_fd, L_fd, E_fl, L_fl, E_cp, L_cp, E_bd, L_bd, E_bl, L_bl = est_cost_POOL_powcycle_intpow(params_exec, params_pres, plat_settings, plat_cost_profile, return_only_breakdown=True)              
            
            if (all_pc): # all power cycles
                total_nvm_read_cost_L = (L_fd + L_fl)*npc
                total_nvm_write_cost_L = (L_bd + L_bl)*npc
                total_nvm_read_cost_E = (E_fd + E_fl)*npc
                total_nvm_write_cost_E = (E_bd + E_bl)*npc
            else:   # single power cycle
                total_nvm_read_cost_L = L_fd + L_fl
                total_nvm_write_cost_L = L_bd + L_bl            
                total_nvm_read_cost_E = E_fd + E_fl
                total_nvm_write_cost_E = E_bd + E_bl
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - POOL dimensions incorrect") 

    elif layer_type == "FC":
        if (common.check_fc(layer)):
            E_rb, L_rb, E_fd_ngt0, L_fd_ngt0, E_fd_n0, L_fd_n0, E_fl, L_fl, E_cp, L_cp, E_bd, L_bd, E_bl, L_bl = est_cost_CONV_powcycle_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, return_only_breakdown=True)                
            
            if (all_pc): # all power cycles
                total_nvm_read_cost_L = (L_fd_ngt0*npc_ngt0) + (L_fd_n0*npc_n0) + (L_fl*npc)
                total_nvm_write_cost_L = (L_bd + L_bl)*npc
                total_nvm_read_cost_E = (E_fd_ngt0*npc_ngt0) + (E_fd_n0*npc_n0) + (E_fl*npc)
                total_nvm_write_cost_E = (E_bd + E_bl)*npc
            else:   # single power cycle
                total_nvm_read_cost_L = L_fd_ngt0 + L_fd_n0 + L_fl
                total_nvm_write_cost_L = L_bd + L_bl            
                total_nvm_read_cost_E = E_fd_ngt0 + E_fd_n0 + E_fl
                total_nvm_write_cost_E = E_bd + E_bl        
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - FC dimensions incorrect") 

    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")


    return total_nvm_read_cost_L, total_nvm_write_cost_L, total_nvm_read_cost_E, total_nvm_write_cost_E







def est_FLOPS_cost_layer(layer, params_exec, params_pres, layer_based_cals):

    if layer['type'] == 'CONV' or layer['type'] == 'FC':
       total_flops, total_macs = est_cost_CONV_flops(layer, params_exec, params_pres, layer_based_cals)
    
    elif layer['type'] == 'BN':
        total_flops, total_macs = est_cost_BN_flops(layer, params_exec, params_pres, layer_based_cals)
    
    elif layer['type'] == 'ADD':
        total_flops, total_macs = est_cost_ADD_flops(layer, params_exec, params_pres, layer_based_cals)
    
    elif layer['type'] == 'POOL':
        total_flops=0; total_macs=0   
        
    elif layer['type'] == 'GAVGPOOL': 
        total_flops, total_macs = est_cost_GAVGPOOL_flops(layer, params_exec, params_pres, layer_based_cals)
    
    elif layer['type'] == 'RELU':
        total_flops=0; total_macs=0
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")

    return total_flops, total_macs
