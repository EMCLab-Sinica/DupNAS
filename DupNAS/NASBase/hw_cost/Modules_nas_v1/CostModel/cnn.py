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
from .conv_tiled import est_cost_CONV_flops
from .pool_tiled import est_cost_GAVGPOOL_flops
from .bn_tiled import est_cost_BN_flops
from .add_tiled import est_cost_ADD_flops


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


def pass_constraint_responsiveness(L_e2e, plat_settings):
    lat_e2e_req = plat_settings['LAT_E2E_REQ']
    if L_e2e == -1 or L_e2e > lat_e2e_req:
        if DEBUG_CONSTRAINTS:
            if L_e2e != -1:
                print(f"Network has latency {L_e2e}, which exceeds latency constraint {lat_e2e_req}")
        return [False, lat_e2e_req, L_e2e]
    else:
        return [True, lat_e2e_req, L_e2e]


############################################################################
# COST ANALYSIS 
############################################################################
# --- SINGLE LAYER ----


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
