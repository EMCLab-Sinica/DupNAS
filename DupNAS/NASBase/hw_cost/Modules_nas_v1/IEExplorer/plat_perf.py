
import math
import sys
import random
import os
from os.path import dirname, realpath
import sys
import csv
import time
import copy
import json 
from typing import Any, Tuple, List


from pprint import pprint
import torch
import numpy as np
import traceback
import numpy as np

from .handler import (
    find_best_solution,
    find_layer_cost_fixed_solution,
    find_layer_flops_fixed_solution,
    find_layer_vm_usage_fixed_solution,
    get_minimum_solution,
)
from ..CostModel.plat_energy_costs import PlatformCostModel
from ..CostModel import common
from ..CostModel.cnn import pass_constraint_storage

#sys.path.append("..")
from ....model.common_types import LAYERTYPES, OPTYPES, Mat
from ....model.common_utils import get_network_obj, get_network_dimension, netobj_to_pyobj

sys.path.append(dirname(dirname(dirname(dirname(dirname(realpath(__file__)))))))
from settings import Settings
#sys.path.append("../..")
#from file_utils import NumpyEncoder


class Settingsplat(Settings):
    def __init__(self):
        self.PLATFORM_SETTINGS = copy.deepcopy(Settings.PLATFORM_SETTINGS)


class PlatPerf:
    def __init__(self, NAS_SETTINGS, PLAT_SETTINGS):

        self.PLAT_SETTINGS = PLAT_SETTINGS        
        self.NAS_SETTINGS = NAS_SETTINGS
        self.PLAT_COST_PROFILE = self.get_cost_model()

    
    # get cost profile per platform (using the benchmark measurements)                
    def get_cost_model(self):
        plat_cost_profile = PlatformCostModel.PLAT_MSP430_EXTNVM
   

        return plat_cost_profile

        

    
    @staticmethod
    def get_inference_latency_verbose(perf_mode_contpow, net_obj, net_config, fixed_params=None):
        
        try:
            sb_blk_choice_key = "<" + ','.join([str(c) for c in net_config]) + ">"                           
                
    
            lat, exec_design_fp, _ = perf_mode_contpow.get_inference_latency(net_obj, fixed_params=fixed_params)
            
 
            subnet_latency_info = { 
                "sn_cpb"  :  net_config,                                 
                "subnet_obj": netobj_to_pyobj(net_obj),
                
             
                "perf_e2e_contpow_fp_lat": lat,
                "perf_exec_design_contpow_fp": exec_design_fp,                     
                            
            } 
                                        
        except Exception as e:
            print("here --")
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc(); print(tb); print("subnet_cpb: ", sb_blk_choice_key)
            #sys.exit()
            subnet_latency_info = None
            
        return subnet_latency_info   
        
    
    
    def get_inference_latency(self, network, fixed_params=None):
        #network = self.get_network(predic_actions, model_fn_type=self.NAS_SETTINGS['MODEL_FN_TYPE']) 
        #network = get_network_obj(subnet)
        error = None
        latency = 0
        network_exec_design = []        

        # exec and pres params not given, so have to find best sol
        if fixed_params == None:
        
            sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE)
            if 'NET_ERROR' in sols:
                print("get_inference_latency error: NET_ERROR !!! ")
                return -1, None, sols['NET_ERROR']['error_code']
                
            
            for each_layer in network:             
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['best_sol'] == None: # cannot find a feasible execution design
                    print("get_inference_latency error: cannot find a feasible execution design")
                    return -1, None, sols[each_layer['name']]['error_code']
                else:
                    latency = latency
                    # network_exec_design.append({'layer' : each_layer['name'], 
                    #                             'alias' : each_layer['alias'],
                    #                             'params': sols[each_layer['name']]['best_sol'][0]['params'],
                    #                             'params_exec': sols[each_layer['name']]['best_sol'][0]['params_exec'],
                    #                             'params_pres': sols[each_layer['name']]['best_sol'][0]['params_pres'],
                    #                             'cost_brk': sols[each_layer['name']]['best_sol'][0]['cost_brk'],
                    #                             'Epc' : sols[each_layer['name']]['best_sol'][0]['Epc'],
                    #                             'Eu' : sols[each_layer['name']]['best_sol'][0]['Eu'],
                    #                             'Le2e' : sols[each_layer['name']]['best_sol'][0]['Le2e'],
                    #                             'npc' : sols[each_layer['name']]['best_sol'][0]['npc'][0],
                    #                             'L_rc_tot' : sols[each_layer['name']]['best_sol'][0]['L_rc_tot'],
                    #                             'vm' : sols[each_layer['name']]['best_sol'][0]['vm'],                                            
                    #                             })
            
            return latency, network_exec_design, error
        
        else:     
            # exec and pres params given
                   
            for lix, each_layer in enumerate(network): 
                cost_stats = find_layer_cost_fixed_solution(each_layer, fixed_params[lix], self.PLAT_SETTINGS, self.PLAT_COST_PROFILE)
                if cost_stats['Le2e'] != None:
                    latency = latency + cost_stats['Le2e']
                    # network_exec_design.append({'layer' : each_layer['name'], 'alias' : each_layer['alias'],
                    #                             'params': fixed_params[lix]['params'],
                    #                             'params_exec': fixed_params[lix].get('params_exec'),
                    #                             'params_pres': fixed_params[lix].get('params_pres'),
                    #                             'cost_brk': cost_stats['cost_brk'],
                    #                             'Epc' : cost_stats['Epc'], 'Eu' : cost_stats['Eu'], 
                    #                             'Le2e' : cost_stats['Le2e'], 'npc' : cost_stats['npc'][0], 
                    #                             'L_rc_tot' : cost_stats['L_rc_tot'],
                    #                             'vm' : cost_stats['vm'],                                            
                    #                             })
                else:
                    print("get_inference_latency error: cost_stats is none")
                    return -1, None, cost_stats['reason']
            
            return latency, network_exec_design, error
            
            
            
    def get_network_flops(self, network,  fixed_params=None, layer_based_cals=False):
        error = None
        network_flops = []
        network_exec_design = []        
        
        # exec and pres params not given, so have to find best sol
        if fixed_params == None and not layer_based_cals:
            sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE)
            for each_layer in network: 
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['best_sol'] == None: # cannot find a feasible execution design
                    print("get_network_flops error: cannot find a feasible execution design")
                    return -1, None, sols[each_layer['name']]['error_code']
                else:
                    solution = {'layer' : each_layer['name'],
                                'alias' : each_layer['alias'],
                                'params': sols[each_layer['name']]['best_sol'][0]['params'],
                                }
                    network_exec_design.append(solution)
                layer_flops = find_layer_flops_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, layer_based_cals=layer_based_cals)
                #print("no sol, layer_flops: ", layer_flops)
                network_flops.append(layer_flops)

            return network_flops, network_exec_design, error

        else:
            # exec and pres params given

            for lix, each_layer in enumerate(network):
                solution = {'layer' : each_layer['name'],
                            'alias' : each_layer['alias'],
                            }
                if not layer_based_cals:
                    solution['params'] = fixed_params[lix]['params']
                else:
                    solution['params'] = None
                layer_flops = find_layer_flops_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE, layer_based_cals=layer_based_cals)
                #print("with sol, layer_flops: ", layer_flops)
                network_exec_design.append(solution)
                network_flops.append(layer_flops)
            return network_flops, network_exec_design, error
        
        


    def get_vm_usage(self, network, fixed_params=None) -> Tuple[bool, List, List, Any]:

        error = None
        network_vm_usage = []
        network_exec_design = []        
        all_layers_fit_vm = True
        
        # exec and pres params not given, so have to find best sol
        if fixed_params == None:
            if self.PLAT_SETTINGS['LAT_E2E_REQ'] > 0:
                # need the best solution for latency constraint
                sols = find_best_solution(network, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE)
            else:
                # if only memory constraints are considered, only the smallest tile size should be checked, not the best one
                sols = get_minimum_solution(network)

            if 'NET_ERROR' in sols:
                return False, [], [], sols['NET_ERROR']['error_code']

            for each_layer in network: 
                # if any one of the layers do not have a best solution, then return -1
                if sols[each_layer['name']]['error_code'] != None: # cannot find a feasible execution design
                    return False, None, None, sols[each_layer['name']]['error_code']
                else:
                    solution = {'layer' : each_layer['name'],
                                'alias' : each_layer['alias'],
                                'params': sols[each_layer['name']]['best_sol'][0]['params'],
                                }
                    network_exec_design.append(solution)
                layer_vm_usage = find_layer_vm_usage_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE)

                cur_layer_fit_vm = layer_vm_usage[0]
                all_layers_fit_vm = all_layers_fit_vm and cur_layer_fit_vm

                network_vm_usage.append(layer_vm_usage)

            return all_layers_fit_vm, network_vm_usage, network_exec_design, error

        else:
            # exec and pres params given

            for lix, each_layer in enumerate(network):
                solution = {'layer' : each_layer['name'],
                            'alias' : each_layer['alias'],
                            }
                solution['params'] = fixed_params[lix]['params']
                network_exec_design.append(solution)
                layer_vm_usage = find_layer_vm_usage_fixed_solution(each_layer, solution, self.PLAT_SETTINGS, self.PLAT_COST_PROFILE)

                cur_layer_fit_vm = layer_vm_usage[0]
                all_layers_fit_vm = all_layers_fit_vm and cur_layer_fit_vm

                network_vm_usage.append(layer_vm_usage)

            return all_layers_fit_vm, network_vm_usage, network_exec_design, error

    def get_nvm_usage(self, network) -> Tuple[bool, List, Any]:
        error = None
        all_layers_fit_nvm, _, network_nvm_usage = pass_constraint_storage(network, self.PLAT_SETTINGS)
        return all_layers_fit_nvm, network_nvm_usage, error

    @classmethod
    def get_latency_info(cls, net_obj, net_config, fixed_params=None):

        global_settings_plat = Settingsplat() # default settings

        performance_model_plat = PlatPerf(global_settings_plat.NAS_SETTINGS_GENERAL, global_settings_plat.PLATFORM_SETTINGS)

        subnet_latency_info = cls.get_inference_latency_verbose(performance_model_plat, net_obj, net_config, fixed_params=fixed_params)

        return subnet_latency_info


# convert exec design into string
def exec_design_to_string(exec_design):

    if exec_design != None:        
        s = ""
        for each_layer in exec_design:        
            #s += ','.join(['='.join(i) for i in each_layer.items()])
            s += str(each_layer)
            s += ","
        return s
    else:
        return ""

