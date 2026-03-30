import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect
import warnings


# local import
from ..CostModel import cnn as cnn
from ..CostModel import common as common 
from ....model.common_types import Mat
from ....file_utils import json_dump, json_load






# with constraints
def explore_full_param_sweep_contpow(layer, plat_settings, plat_cost_profile, report_topN=0.5, best_selection='first'):
    
    R = layer['OFM'].h; C = layer['OFM'].w; M = layer['OFM'].ch; N = layer['IFM'].ch
    H = layer['IFM'].h; W = layer['IFM'].w
    Kh = layer['K'].h; Kw = layer['K'].w
    stride = layer['stride']

    # get search space
    tr_lst, tc_lst, tm_lst, tn_lst = common.filter_legal_tilesizes(None, None, None, None, H, W, R, C, M, N, layer_type = layer['type'], op_type=layer['optype'])
    inter_tile_order = common.filter_legal_reuseschems(layer_type = layer['type'], op_type=layer['optype'])
    

    result_pass = []
    result_fail = []    
    search_space_size = 0

    # -- all tile size permutations
    for Tr in tr_lst:
        for Tc in tc_lst:
            #Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride = layer['stride'], layer_type = layer['type'])
            Tri=H
            Tci=W
            for Tm in tm_lst:
                for Tn in tn_lst:

                    # -- all loop orders
                    for inter_lo in inter_tile_order:                        
                        #print (R, C, M, N, Tr, Tc, Tm, Tn, inter_lo)
                        
                            search_space_size+=1

                            # -- check if passes the initial constraints ?                                                        
                            params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': inter_lo}
                            #print("H, W, R, C, M, N = ", H, W, R, C, M, N)
                            params_pres = {'backup_batch_size': 1} # using S=1 for constant 
                            #print('tile_size',[Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo',inter_lo)                          
                            res_cons_c0 = cnn.pass_constraint_spatial(layer, plat_settings, params_exec, params_pres); common.check_infnan(res_cons_c0)  
                            #print(res_cons_c0)
                            if (res_cons_c0[0]):
                                result_pass.append({
                                        'params' : common.to_string_params_all(params_exec, params_pres),                                                                                
                                        'params_exec': params_exec,
                                        'params_pres': params_pres,
                                        'Epc': [None, None], 'Le2e': None, 
                                        'Lpc': [None, None], 'Lrc': [None, None], 'Eu' : None, 'npc' : [None, None, None],
                                        'L_rc_tot' : None,
                                        'cost_brk': None,
                                        'vm' : res_cons_c0[2]
                                })

                            else:
                                #lay_E = 0
                                #lay_L = 0
                                result_fail.append({
                                    'params' : common.to_string_params_all(params_exec, None),                                                                                
                                    'reason': 'FAILED_C0',
                                    'Epc': [None, None], 'Le2e': None,
                                    'npc' : [None, None, None],     
                                    'cost_brk': None, 'L_rc_tot' : None,                                                      
                                    'vm' : res_cons_c0[2]
                                })                              
                                
    
    #print("Layer [%s] eval. complete. PASS= %d/%d = %.1f" % (layer['name'], len(result_pass), search_space_size, (len(result_pass)/search_space_size)*100.0 ))

    if (len(result_pass) > 0):
        # -- find best solution (lowest E2E latency)  
        min_lat = 0 # np.min([r['Le2e'] for r in result_pass])
        all_best_sols = [r for r in result_pass if r['Le2e'] == min_lat]
        best_solution = sorted(all_best_sols, key = lambda i: i['vm'])

        # sorted_result_pass = sorted(result_pass, key = lambda i: i['Le2e'])
        
        # # top N% solutions
        # nperc = report_topN
        # nnum = int(np.ceil(nperc* len(sorted_result_pass)))
        # pass_topN = sorted_result_pass[0:nnum]

        # # sort and save the top N% failed solutions    
        # sorted_results_fail_c0 = sorted([f for f in result_fail if f['reason'] == 'FAILED_C0'], key = lambda i: i['vm'])
        # nnum = int(np.ceil(nperc * len(sorted_results_fail_c0)))
        # sorted_results_fail_c0 = sorted_results_fail_c0[0:nnum]
        # sorted_results_fail_c1 = []

        return best_solution, [], [], [], []
    
    else:

        warnings.warn("WARNING: Layer [%s] - unable to find a solution" % (layer['name']))
        best_solution = None
        result_pass = []
        sorted_results_fail_c0 = []
        sorted_results_fail_c1 = []
        pass_topN = []
        
        return best_solution, result_pass, sorted_results_fail_c0, sorted_results_fail_c1, pass_topN


# get end-to-end latency for a specific fixed param
def get_le2e_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile):
    result = {}
    res_cons_c0 = cnn.pass_constraint_spatial(layer, plat_settings, params_exec, params_pres)
    if (res_cons_c0[0]):        
        result = {
            'params' : common.to_string_params_all(params_exec, params_pres),                                                                                
            'Epc': [None, None], 'Le2e': None, 
            'Lpc': [None, None], 'Lrc': [None, None], 'Eu' : None, 'npc' : [1, 1, 0],
            'L_rc_tot' : None,
            'cost_brk': None,
            'vm' : res_cons_c0[2]
        }
    else:
        result = {
                'params' : common.to_string_params_all(params_exec, params_pres),                                        
                'reason': 'FAILED_C0',
                'Epc': [None, None], 'Le2e': None, 'Lpc': [None, None],
                'npc' : [None, None, None],
                'L_rc_tot': None,
                'cost_brk': None,
                'vm' : res_cons_c0[2]
            }

    return result




def get_flops_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, layer_based_cals):
    total_flops, total_macs = cnn.est_FLOPS_cost_layer(layer, params_exec, params_pres, layer_based_cals)
    return total_flops, total_macs


def get_vm_usage_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile):
    vm_usage = cnn.pass_constraint_spatial(layer, plat_settings, params_exec, params_pres)
    return vm_usage
