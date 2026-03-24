import sys
import os.path
import itertools
from pprint import pprint
from .train_supernet import run_supernet_train
import statistics
import traceback
from typing import Dict, List, Tuple
import numpy as np
import torch
import onnx
import re, ast

from settings import Settings, SSOptPolicy, Stages, arg_parser
from NASBase import file_utils, utils
from NASBase.evo_search.search import evo_search
from NASBase.model.common_utils import get_supernet, parametric_supernet_choices, parametric_supernet_blk_choices
from NASBase.ss_optimization.ss_opt import ss_optimization
from NASBase.fine_tune import fine_tune_best_solution
from logger.remote_logger import get_remote_logger_obj, get_remote_logger_basic_init_params
from NASBase.ss_optimization.subnet_utils import sample_subnet_configs_from_file, check_constraints, merge_constraint_stats

from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.model.common_utils import get_network_dimension, get_network_obj, netobj_to_pyobj, get_supernet, get_dummy_net_input_tensor


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


supernet_choices, _ = parametric_supernet_choices(global_settings=Settings)
#supernet_choices = [[0.5, 128]]
supernet_block_choices = parametric_supernet_blk_choices(global_settings=Settings)

dataset = Settings.NAS_SETTINGS_GENERAL['DATASET']
first_block_hard_coded=Settings.NAS_SETTINGS_PER_DATASET[dataset]['FIRST_BLOCK_HARD_CODED']

def load_cpbs_from_txt(path):
    """
    Parse every line that contains 'cpb: [[...]]' and return [(name, cpb), ...].
    No filtering. 'name' may be None if not found.
    """

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        msg = f"[ERROR] CPB file not found: {os.path.abspath(path)}"
        print(msg)
        return []


    out = []
    pat_cpb  = re.compile(r'cpb:\s*(\[\[.*?\]\])')
    pat_name = re.compile(r'subnet:\s*([^\s,]+)')
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m_cpb = pat_cpb.search(line)
            if not m_cpb:
                continue
            cpb = ast.literal_eval(m_cpb.group(1))  # -> list-of-lists
            m_name = pat_name.search(line)
            name = m_name.group(1) if m_name else None
            out.append((name, cpb))
    return out


#specific_super_net = [[0.1, 32]]

for width_multiplier, input_resolution in supernet_choices:
    
    cpb_path = f"pdq_bal_w{width_multiplier}_ir{input_resolution}_check_per_supernet.txt"
    cpb_tuples = load_cpbs_from_txt(cpb_path)

    if not cpb_tuples:
        print(f"[WARN] No CPB entries found (or file missing): {os.path.abspath(cpb_path)}")
        # Option A: skip this supernet
        continue
    else:
        print("number of subnet: ", len(cpb_tuples))

    supernet = get_supernet(global_settings=Settings, dataset=dataset, width_multiplier=width_multiplier)

    # build subnet configs from file (all of them)
    all_subnet_configs_from_file = list(
        sample_subnet_configs_from_file(supernet, cpb_tuples, first_block_hard_coded=first_block_hard_coded))

    print(f"[INFO] Loaded {len(all_subnet_configs_from_file)} subnets from {cpb_path}")
    
    #len_all_subnet_configs = list(all_subnet_configs_lst_generator)
    #print("Total subnet configs:", len(len_all_subnet_configs))
    
    needs_subnets = Settings.NAS_SSOPTIMIZER_SETTINGS['SUBNET_SAMPLE_SIZE']
    all_subnet_configs = list(itertools.islice(all_subnet_configs_from_file, needs_subnets))
    #all_subnet_configs = list(all_subnet_configs_lst_generator)

    net_input = get_dummy_net_input_tensor(Settings, input_resolution)
    print("width_multiplier, input_resolution :", width_multiplier, input_resolution)
    
    for i, each_subnet_config in enumerate(all_subnet_configs):        
            
            each_subnet = MNASSubNet(**each_subnet_config)
            #print("i: ", i)
            under_mem = False
            #print(each_subnet)
            
            try:        
                subnet_name = each_subnet.name
                subnet_cpb = each_subnet.choice_per_block
                #print("subnet_name: ", subnet_name)
                print("subnet_cpb: ", subnet_cpb)
                # -- get subnet costs
                subnet_dims = get_network_dimension(each_subnet, input_tensor = net_input)
                #print ("dims: ", subnet_dims)         
                subnet_obj = get_network_obj(subnet_dims)
                #print("subnet_obj: ",subnet_obj)
                #continue
                
                vm_available = Settings.NAS_SETTINGS_GENERAL['VMSIZE']
                print("Convert pth to onnx:")
                onnx_path = Settings.NAS_EVOSEARCH_SETTINGS['SAMPLE_ONNX_FILE_PATH']
                model_name = 'w'+str(width_multiplier)+'_ir'+str(input_resolution)+'_'+str(subnet_name)
                input_names = ["input"]
                output_names = ["output"]

                onnx_file = os.path.join(onnx_path, model_name + '.onnx')

                try:
                        # Export the ONNX model
                    torch.onnx.export(
                        each_subnet,
                        net_input,
                        onnx_file,
                        input_names=input_names,
                        output_names=output_names,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        #dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
                    )
                except Exception as e:
                    print(f"[*] Failed to export ONNX: {onnx_file}")
                    print(f"[*] Error: {e}")
                    continue  # skip this model

                if not os.path.exists(onnx_file) or os.path.getsize(onnx_file) < 10000:
                    print(f"[*] ONNX file not created properly or is too small: {onnx_file}")
                    continue

                try:
                    onnx_model = onnx.load(onnx_file)
                    onnx.checker.check_model(onnx_model)
                
                except Exception as e:
                    print(f"[*] ONNX file load/check failed: {onnx_file}")
                    print(f"[*] Error: {e}")
                    continue

            except Exception as e:            
                error_net_perf = True
                pprint(e)
                tb = traceback.format_exc()
                print(tb)
                print("subnet_cpb: ", subnet_cpb)
