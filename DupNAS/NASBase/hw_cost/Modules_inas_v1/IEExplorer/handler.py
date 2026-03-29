import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect

#sys.path.append('..')
from ..CostModel import cnn as cnn
from ..CostModel import common as common 
from .explore_intpow import get_energy_all_params_intpow, explore_full_param_sweep_intpow, get_le2e_fixed_params_intpow, get_flops_fixed_params_intpow, get_data_access_layer_intpow, get_vm_usage_fixed_params_intpow
from .explore_contpow import get_energy_all_params_contpow, explore_full_param_sweep_contpow, get_le2e_fixed_params_contpow, get_flops_fixed_params_contpow, get_vm_usage_fixed_params_contpow



############################################################################
# HELPERS
############################################################################
class IEErrorCodes():
    IE_ERRORCODE_NONE = None
    IE_ERRORCODE_SPATIAL_C0 = 11
    IE_ERRORCODE_ATOMICITY_C1 = 12
    IE_ERRORCODE_STORAGE_C2 = 13
    IE_ERRORCODE_UNKNOWN = -1
    
    

# check what the exec design error is 
def _check_error(fail_solutions_c0, fail_solutions_c1, fail_solutions_c2):
    if (len(fail_solutions_c1) > 0):
        return IEErrorCodes.IE_ERRORCODE_ATOMICITY_C1    
    elif (len(fail_solutions_c0) > 0):
        return IEErrorCodes.IE_ERRORCODE_SPATIAL_C0    
    elif fail_solutions_c2:
        return IEErrorCodes.IE_ERRORCODE_STORAGE_C2
    else:
        return IEErrorCodes.IE_ERRORCODE_UNKNOWN




# check if the layer dimensions are valid
def _check_layer(layer):
    if layer['type'] == "CONV":
        if not common.check_conv(layer):
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - CONV dimensions incorrect")
    elif (layer['type'] == "POOL") or (layer['type'] == "GAVGPOOL"):
        if not common.check_pool(layer):
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - POOL dimensions incorrect")
    elif layer['type'] == "FC":
        if not common.check_fc(layer):   # assumes the FC layer is formulated as a CONV layer                
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - FC dimensions incorrect")
    elif (layer['type'] == "BN"): 
        if not common.check_bn(layer):
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - BN dimensions incorrect")    
    elif (layer['type'] == "RELU"):
        if not common.check_relu(layer):
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - RELU dimensions incorrect")        
    elif (layer['type'] == "ADD"):
        if not common.check_add(layer):
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - ADD dimensions incorrect")        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type, " + layer['type'])



############################################################################
# Query E2E latency / cost
############################################################################
# given a specific exec + pres solution, find end-to-end latency
def find_layer_cost_fixed_solution(layer, solution, plat_settings, plat_cost_profile, power_type='INT'):
    cost_stats_per_layer = None    
        
    #print ("------- Finding Lat E2E for : %s" % layer['name'])
    _check_layer(layer)
    if solution['params'] != None:
        # find parameters of the solution
        Kh = layer['K'].h; Kw = layer['K'].w
        stride = layer['stride']
        Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(solution['params'])
        Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride = layer['stride'], layer_type = layer['type'])
        params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': reuse_sch}    
        params_pres = {'backup_batch_size': S}   
        print(params_exec)
        
        # find cost stats
        if power_type == 'INT':
            cost_stats = get_le2e_fixed_params_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile)              
        elif power_type == 'CHK_PASS_ATOMICITY':
            cost_stats = get_le2e_fixed_params_contpow(layer,  params_exec, params_pres, plat_settings, plat_cost_profile)              
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown power type, " + power_type)
        
        return cost_stats
        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - solution is None")
    

def get_layer_data_access_cost(layer, solution, plat_settings, plat_cost_profile, power_type='INT'):
    cost_per_layer = {}    
    _check_layer(layer)
    if solution['params'] != None:
        Kh = layer['K'].h; Kw = layer['K'].w
        stride = layer['stride']
        Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(solution['params'])
        Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw)
        params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': reuse_sch}
        params_pres = {'backup_batch_size': S}

        tot_data_rd_cost_L, tot_data_wr_cost_L, tot_data_rd_cost_E, tot_data_wr_cost_E = get_data_access_layer_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile)

        return tot_data_rd_cost_L, tot_data_wr_cost_L, tot_data_rd_cost_E, tot_data_wr_cost_E
    
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - solution is None")


        
def find_layer_imc_fixed_solution():
    # TODO
    raise NotImplementedError

def find_layer_flops_fixed_solution(layer, solution, plat_settings, plat_cost_profile, power_type='INT', layer_based_cals=False):
    # follow the same format of find_layer_cost_fixed_solution above
    _check_layer(layer)

    if solution['params'] != None or layer_based_cals:
        Kh = layer['K'].h; Kw = layer['K'].w
        if not layer_based_cals:
            Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(solution['params'])
            Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride = layer['stride'], layer_type = layer['type'])
        else:
            Tr = Tc = Tm = Tn = reuse_sch = S = None
            Tri = Tci = None
        params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': reuse_sch}
        params_pres = {'backup_batch_size': S}
        if power_type == 'INT':
            flops_stats, _ = get_flops_fixed_params_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, layer_based_cals)
        elif power_type == 'CONT':
            flops_stats, _ = get_flops_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile, layer_based_cals)
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown power type, " + power_type)

        return flops_stats

    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - solution is None")
        
def find_layer_vm_usage_fixed_solution(layer, solution, plat_settings, plat_cost_profile, power_type='INT'):
    # follow the same format of find_layer_cost_fixed_solution above
    _check_layer(layer)

    if solution['params'] != None:
        Kh = layer['K'].h; Kw = layer['K'].w
        Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(solution['params'])
        Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride = layer['stride'], layer_type = layer['type'])
        params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': reuse_sch}
        params_pres = {'backup_batch_size': S}
        if power_type == 'INT':
            vm_usage_stats = get_vm_usage_fixed_params_intpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile)
        elif power_type == 'CONT':
            vm_usage_stats = get_vm_usage_fixed_params_contpow(layer, params_exec, params_pres, plat_settings, plat_cost_profile)
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown power type, " + power_type)

        return vm_usage_stats

    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - solution is None")


############################################################################
# Explorer Handlers
############################################################################
def find_energy_all_solutions(network, plat_settings, plat_cost_profile):
    sol_per_layer = []    
    for each_layer in network: 
        _check_layer(each_layer)

        all_sols = get_energy_all_params_intpow(each_layer, plat_settings, plat_cost_profile)                
        sol_per_layer[each_layer['name']] = {
            'all_sols': all_sols,
        }
    
    return sol_per_layer



def find_best_solution(network, plat_settings, plat_cost_profile, power_type='CONT', cost_breakdown=False):
    #print("------- Finding the best solution...")

    sol_per_layer = {}    

    res_cons_c2 = cnn.pass_constraint_storage(network, plat_settings)
    if (not res_cons_c2[0]):
        # network level constraint failed
        sol_per_layer['NET_ERROR'] = {            
            'error_code' : _check_error([], [], not res_cons_c2[0]),
        }
        print("find_best_solution: network level constraint failed")
        return sol_per_layer
    
    
    for each_layer in network: 
        #print("processing - ", each_layer['alias'], each_layer['lcnt'])
        #print ("------- Exploring : %s" % each_layer['name'])
        _check_layer(each_layer)

        best_solution=None; pass_solutions=None; fail_solutions=None
        if power_type == 'INT': # INT power
            best_solution, pass_solutions, fail_solutions_c0, fail_solutions_c1, pass_topN = explore_full_param_sweep_intpow(each_layer, plat_settings, plat_cost_profile)                
        elif power_type == 'CONT': # CONT power
            best_solution, pass_solutions, fail_solutions_c0, fail_solutions_c1, pass_topN = explore_full_param_sweep_contpow(each_layer, plat_settings, plat_cost_profile)                
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - Unknown power type "+power_type)
                
        if best_solution == None or not res_cons_c2[0]:   # no solution found                        
            sol_per_layer[each_layer['name']] = {
                'best_sol': best_solution,            
                #'pass_sols': pass_solutions,
                'pass_topN' : pass_topN,
                'fail_c0_topN': fail_solutions_c0,
                'fail_c1_topN': fail_solutions_c1,
                'fail_c2': [res_cons_c2[2]],
                'error_code' : _check_error(fail_solutions_c0, fail_solutions_c1, not res_cons_c2[0]),
            }
            print(each_layer['name'], "find_best_solution: no solution found")
            return sol_per_layer
        else:
            sol_per_layer[each_layer['name']] = {
                'best_sol': best_solution,
                #'pass_sols': pass_solutions,
                'pass_topN' : pass_topN,
                'fail_c0_topN': fail_solutions_c0,
                'fail_c1_topN': fail_solutions_c1,
                'error_code' : None,
            }
            #print(each_layer['name'], "find_best_solution: successed")
    
    return sol_per_layer


def get_minimum_solution(network):
    sol_per_layer = {}
    for each_layer in network:
        _check_layer(each_layer)

        Kh = each_layer['K'].h; Kw = each_layer['K'].w
        stride = each_layer['stride']

        Tr = Tc = Tm = Tn = 2
        S = 1
        inter_lo = 'reuse_I'  # does not really matter for S = 1

        Tri, Tci = common._calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, layer_type = each_layer['type'], stride=stride)

        params_exec = {'tile_size': [Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn], 'inter_lo': inter_lo}
        params_pres = {'backup_batch_size': S}

        minimum_solutions = [{
            'params' : common.to_string_params_all(params_exec, params_pres),
        }]

        sol_per_layer[each_layer['name']] = {
            'best_sol': minimum_solutions,
            'error_code' : None,
        }

    return sol_per_layer
