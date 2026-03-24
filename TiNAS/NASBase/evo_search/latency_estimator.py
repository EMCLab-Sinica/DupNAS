#import yaml
#from ofa.utils import download_url

import sys, os
from pprint import pprint
import torch
import numpy as np
import traceback
import inspect
import math

from settings import Settings, arg_parser, load_settings


#sys.path.append("..")
#from NASBase.model.mnas_arch import MNASSuperNet, MNASSubNet

if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mobile':
    from NASBase.model.mnas_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.mnas_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mbv2':
    from NASBase.model.mbv2_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.mbv2_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    from NASBase.model.shuffle_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.shuffle_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    from NASBase.model.inception_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.inception_ss import *



from NASBase.model.common_utils import (
    get_network_dimension, get_network_obj, get_subnet, parametric_supernet_blk_choices, blkchoices_to_blkchoices_ixs, get_subnet_from_config,
    get_dummy_net_input_tensor,
)
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf

#sys.path.append("../..")
from NASBase import utils as utils
from NASBase import file_utils as file_utils
#from NASBase.evo_search.evolution_finder import op_duplication_subnet

if Settings.NAS_SETTINGS_GENERAL['MODE'] == 'none': 
    USE_TS = False
else:
    USE_TS = True





class LatencyEstimator(object):

    def __init__(self, global_settings):
        self.global_settings = global_settings
        self.dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']
        
    
    def _get_net_perf(self, subnet_obj, settings_inst, fixed_params=None, power_type='CONT'):        
        #print("get_network_perf::Enter")
        performance_model = PlatPerf(settings_inst.NAS_SETTINGS_GENERAL, settings_inst.PLATFORM_SETTINGS)
        time_performance, exec_design, error = performance_model.get_inference_latency(subnet_obj, fixed_params=fixed_params, power_type=power_type)
        
        #subnet_cust_net_obj = 
        return time_performance, exec_design, error   
        
    
    # get both latency under cont/int pow
    def predict_network_latency(self, net_config, net_obj, input_size, input_ch):
        #print("predict_network_latency:: Enter subnet_config {}".format(net_config))
        
            
        net_input = get_dummy_net_input_tensor(self.global_settings, input_size)
                    
        try:     
            sb_blk_choice_key = "<" + ','.join([str(c) for c in net_config]) + ">"                           
            
            # -- get subnet costs
            subnet_dims = get_network_dimension(net_obj, input_tensor = net_input)         
            subnet_obj = get_network_obj(subnet_dims)       
                        
            # get perf for INT pow            
            #e2e_lat_intpow, exec_design_intpow, error_intpow = self._get_net_perf(subnet_obj, self.global_settings, power_type='INT')
                        
            #ip_tot_npc = np.sum([l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1        # total power cycles
            #ip_tot_rc = np.sum([l['L_rc_tot'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1    # total recharge time
            #ip_tot_rb = np.sum([l['cost_brk']['rb'][1] * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
                                    
            # get perf for CONT pow                 
            # cont pow performance - fixed params : same as intpow
            #print(net_config)
           # print(net_obj)
            # if USE_TS:
            #     if self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'pdq':
            #         under_mem, oripeak, peak_for_all, total_pdq_config, ori_mac, plus_mac, ori_access, plus_access, max_features, cached_nvm_fit  = op_duplication_subnet(self.global_settings, net_input, net_obj)

            #     elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinyts':
            #         under_mem, oripeak, peak_after_sp, micrographs, ori_mac, plus_mac, ori_access, plus_access, max_features, cached_nvm_fit = op_duplication_subnet(self.global_settings, net_input, net_obj)

            #     elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinynas':  ## wait for new version
            #         under_mem, oripeak, peak_after_patch, min_mem_result, ori_mac, plus_mac, ori_access, plus_access, max_features, cached_nvm_fit = op_duplication_subnet(self.global_settings, net_input, net_obj)
                

            e2e_lat_contpow_fp, exec_design_contpow_fp, error_contpow_fp = self._get_net_perf(subnet_obj, self.global_settings, power_type='CONT')                        

            # error reporting            
            #if any(x == -1 for x in [ip_tot_npc, ip_tot_rc, ip_tot_rb, e2e_lat_intpow]):                
            #    error_net_perf = True
            #    int_mng_cost_proportion_cpfp = -1
            #else:
             #   error_net_perf = False                

                # Calc IMC (vs. contpow_fp)
                #active_time = (e2e_lat_intpow - ip_tot_rc)            
                #int_mng_cost_proportion_cpfp = ((active_time-e2e_lat_contpow_fp)/active_time)*100
                
            subnet_latency_info = {                        
                
                # perf int pow 
               # "error_net_perf"  : error_net_perf,          
                #"perf_e2e_intpow_lat": e2e_lat_intpow,
                #"perf_exec_design_intpow": exec_design_intpow,
                #"perf_error_intpow": error_intpow,
                
                # perf cont pow - fixed params
                "perf_e2e_contpow_fp_lat": e2e_lat_contpow_fp,
                "perf_exec_design_contpow_fp": exec_design_contpow_fp,
                "perf_error_contpow_fp": error_contpow_fp,          
                            
                # proportion
                #"imc_prop" :  int_mng_cost_proportion_cpfp,
            }        
                                            
        except Exception as e:            
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc(); print(tb); print("subnet_cpb: ", sb_blk_choice_key)                        
        
        return subnet_latency_info

    
    
    
    def predict_efficiency(self, net_config, supernet_config, input_ch):
        width_multiplier, input_resolution = supernet_config

        pow_type = self.global_settings.PLATFORM_SETTINGS['POW_TYPE']
        # supernet = get_supernet(self.global_settings, self.dataset, load_state=False)        
        blk_choices = parametric_supernet_blk_choices(global_settings=self.global_settings)
        # get_subnet() wants block choice *indices*
        subnet_choice_per_blk = blkchoices_to_blkchoices_ixs(blk_choices, net_config)
        subnet = get_subnet(self.global_settings, self.dataset, blk_choices, subnet_choice_per_blk, -1, width_multiplier, input_resolution)
        net_latency_info = self.predict_network_latency(net_config, subnet, input_resolution, input_ch)
        
        if pow_type == 'CONT':
            lat = net_latency_info['perf_e2e_contpow_fp_lat']
        #elif pow_type == 'INT':
        #    lat = net_latency_info['perf_e2e_intpow_lat']
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown pow_type, " + pow_type)

        #imc = net_latency_info['imc_prop']
        
        return lat
        
    def predict_nvm_usage(self, net_config, supernet_config,):
        performance_model = PlatPerf(self.global_settings.NAS_SETTINGS_GENERAL, self.global_settings.PLATFORM_SETTINGS)

        subnet_obj, _ = get_subnet_from_config(self.global_settings, self.dataset, net_config, supernet_config)

        all_layers_fit_nvm, _, _ = performance_model.get_nvm_usage(subnet_obj)

        return all_layers_fit_nvm

    def predict_network_latency_verbose(self, subnet_choice_per_blk, supernet_config):
        performance_model = PlatPerf(self.global_settings.NAS_SETTINGS_GENERAL, self.global_settings.PLATFORM_SETTINGS)

        subnet_obj, _ = get_subnet_from_config(self.global_settings, self.dataset, subnet_choice_per_blk, supernet_config)

        subnet_latency_info = performance_model.get_latency_info(subnet_obj, subnet_choice_per_blk)

        return subnet_latency_info
