import os, datetime
import itertools
from pprint import pprint
import statistics
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
import onnx, shutil, tempfile

from settings import Settings
from NASBase.hw_cost.Modules_nas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.model.common_utils import get_network_dimension, get_network_obj, netobj_to_pyobj, get_supernet, get_dummy_net_input_tensor
from NASBase.multiprocessing_helper import get_max_num_workers, run_multiprocessing_workers
from NASBase.ss_optimization.subnet_utils import sample_subnet_configs, check_constraints, merge_constraint_stats
from NASBase.duplication.dup_module import model_tracing

if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mbv2':
    from NASBase.model.mbv2_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.mbv2_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    from NASBase.model.shuffle_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.shuffle_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    from NASBase.model.inception_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.inception_ss import *


NetChoice = Tuple[float, float]
if Settings.NAS_SETTINGS_GENERAL['MODE'] == 'none': 
    USE_TS = False
else:
    USE_TS = True

USE_LATENCY_PROXY = False


if USE_LATENCY_PROXY:
    LC_RATIO = Settings.NAS_EVOSEARCH_SETTINGS['LATENCY_RATIO']
    LC_PROXY = Settings.NAS_EVOSEARCH_SETTINGS['LATENCY_PROXY']


def safe_remove(file):
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception as rr:
            print(f"[WARN] Failed to remove {file}: {rr}")

def flops_worker(cpuid, global_settings: Settings, width_multiplier, input_resolution, layer_based_cals, subnet_config_list):
    print("CPUID [%d] :: Enter : has %d jobs " % (cpuid, len(subnet_config_list)))
    
    print("Starting processes for width_multiplier={}, input_resolution={}..".format(width_multiplier, input_resolution))
    
    # init
    subnet_name="UNKNOWN_SUBNET";subnet_cpb=[];subnet_obj=[]
    
    net_input = get_dummy_net_input_tensor(global_settings, input_resolution)

    subnet_results = []
    error_net_perf = False
    check_per_results = []
    per_result = {'id': None, 'oripeak': 0, 'mem': 0, 'flops': 0, 'nvm': 0, 'under_nvm': False ,'under_vm': False, 'cpb': None, 'TS': False}

    performance_model = PlatPerf(global_settings.NAS_SETTINGS_GENERAL, global_settings.PLATFORM_SETTINGS)

    constraint_stats = {'flops_max': float('-inf'), 'flops_min': float('inf')}

    network_flops_contpow = []

    for i, each_subnet_config in enumerate(subnet_config_list):        
        
        each_subnet = MNASSubNet(**each_subnet_config)
        print("i: ", i)
        under_mem = False
        flops_sum = 0
        mac_count = 0
        per_result = {'id': None, 'oripeak': 0, 'mem': 0, 'flops': 0, 'nvm': 0, 'under_nvm': False ,'under': False, 'ori_mac': 0, 'plus_mac': 0, 'ori_access': 0, 'plus_access': 0, 'cpb': None, 'TS': False}
        subnet_latency_info = {}
        
        #print(each_subnet)
        try:        
            subnet_name = each_subnet.name
            subnet_cpb = each_subnet.choice_per_block
            #print("subnet_name: ", subnet_name)
            #print("subnet_cpb: ", subnet_cpb)
            # -- get subnet costs
            subnet_dims = get_network_dimension(each_subnet, input_tensor = net_input)
            #print ("dims: ", subnet_dims)         
            subnet_obj = get_network_obj(subnet_dims)
            #print("subnet_obj: ",subnet_obj)
            per_result['id'] = subnet_name
            per_result['cpb'] = subnet_cpb
            
            
            if USE_TS:
                vm_available = global_settings.NAS_SETTINGS_GENERAL['VMSIZE']
                print("Convert pth to onnx:")
                each_subnet.eval()

                onnx_path = global_settings.NAS_EVOSEARCH_SETTINGS['ONNX_FILE_PATH']
                model_name = f'sso_sample_cpu{cpuid}_{i}'
                input_names = ["input"]
                output_names = ["output"]
                under_mem = False
                os.makedirs(onnx_path, exist_ok=True)
                onnx_file = os.path.join(onnx_path, model_name + '.onnx')
                onnx_file_inf = os.path.join(onnx_path, model_name + '_inferred.onnx')

                # Clean up any stale file from previous attempts
                if os.path.exists(onnx_file):
                    try:
                        os.remove(onnx_file)
                    except OSError:
                        pass
                        
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
                    # make sure we don't leave behind a partial file
                    if os.path.exists(onnx_file):
                        try:
                            os.remove(onnx_file)
                        except OSError:
                            pass
                    check_per_results.append(per_result.copy())
                    continue  # skip this subnet

                if (not os.path.exists(onnx_file)) or (os.path.getsize(onnx_file) < 10000):
                    print(f"[*] ONNX file not created properly or is too small: {onnx_file}")
                    # cleanup to avoid future confusion
                    if os.path.exists(onnx_file):
                        try:
                            os.remove(onnx_file)
                        except OSError:
                            pass
                    continue

                try:
                    onnx_model = onnx.load(onnx_file)
                    onnx.checker.check_model(onnx_model)
                
                except Exception as e:
                    print(f"[*] ONNX file load/check failed: {onnx_file}")
                    print(f"[*] Error: {e}")
                    # cleanup to avoid later DecodeError on stale file
                    try:
                        os.remove(onnx_file)
                    except OSError:
                        pass
                    check_per_results.append(per_result.copy())
                    continue

                #---- call model_tracing by onnx ----#

                if global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'dupnas':
                    try:
                        which_over_mem, oripeak, total_latency, under_mem, total_pdq_config, duration, time_record, remaining_peaks, ori_mac, plus_mac, ori_access, plus_access, max_mem_without_dup = model_tracing(onnx_path, model_name)
                    finally:
                        safe_remove(onnx_file)
                        safe_remove(onnx_file_inf)

                    per_result['oripeak'] = oripeak
                    per_result['ori_mac'] = ori_mac
                    per_result['plus_mac'] = plus_mac
                    per_result['ori_access'] = ori_access
                    per_result['plus_access'] = plus_access

                    if len(which_over_mem)>0:
                        per_result['TS'] = True
                    else:
                        per_result['TS'] = False 
                    
                    if not under_mem:
                        print("can not find valid duplication for all peak nodes")
                        per_result['under'] = False
                        check_per_results.append(per_result.copy())
                        continue

                    if USE_LATENCY_PROXY:
                        total_latency = (ori_mac + plus_mac)+ LC_RATIO * (ori_access + plus_access)
                        if total_latency > LC_PROXY:
                            print("Over latency constraint")
                            under_mem = False
                            per_result['under'] = False
                            check_per_results.append(per_result.copy())
                            continue

                    peak_after_dup = 0
                    if total_pdq_config:        
                        for idx, entry in enumerate(total_pdq_config):
                            if entry['after_peak_mem'] > peak_after_dup:
                                peak_after_dup = entry['after_peak_mem']
                    
                    if max_mem_without_dup:
                        peak_for_all = max(max_mem_without_dup, peak_after_dup)

                    per_result['mem'] = peak_for_all
                    
                    if peak_for_all > (vm_available*1024):
                        print("after TS, still over vm")
                        under_mem = False
                        per_result['under'] = False
                        check_per_results.append(per_result.copy())
                        continue
                    else:
                        under_mem = True
                        per_result['under'] = True
                

                elif global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinyts':
                    try:
                        which_over_mem, oripeak, duplicated_per_micro, split_ts_per_micro, micrographs, peak_mem_per_micro, inner_time, unsplittable_op_indices, maxone_unsp, ori_mac, plus_mac, ori_access, plus_access = model_tracing(onnx_path, model_name)
                    finally:
                        safe_remove(onnx_file)
                        safe_remove(onnx_file_inf)

                    per_result['oripeak'] = oripeak
                    per_result['ori_mac'] = ori_mac
                    per_result['plus_mac'] = plus_mac
                    per_result['ori_access'] = ori_access
                    per_result['plus_access'] = plus_access

                    
                    if len(which_over_mem)>0:
                        per_result['TS'] = True
                    else:
                        per_result['TS'] = False

                    peak_after_sp=0
                    if micrographs:
                        for m_id, micro in enumerate(micrographs):
                            micro_peak, micro_mode, micro_sp = peak_mem_per_micro[m_id]

                            if micro_peak > peak_after_sp:
                                peak_after_sp = micro_peak
                    
                    if maxone_unsp > peak_after_sp:
                        peak_after_sp = maxone_unsp

                    per_result['mem'] = peak_after_sp

                    if USE_LATENCY_PROXY:
                        total_latency = (ori_mac + plus_mac)+ LC_RATIO * (ori_access + plus_access)
                        if total_latency > LC_PROXY:
                            print("Over latency constraint")
                            under_mem = False
                            per_result['under'] = False
                            check_per_results.append(per_result.copy())
                            continue

                    if peak_after_sp > (vm_available*1024):
                        print("after tinyts, can not meet vm constraint")
                        under_mem = False
                        per_result['under'] = False
                        check_per_results.append(per_result.copy())
                        continue
                    else:
                        under_mem = True
                        per_result['under'] = True

                elif global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'patchts':  ## wait for new version
                    try:
                        which_over_mem, oripeak, min_mem_result, min_latency_result, tried_but_not_valid = model_tracing(onnx_path, model_name) 
                    finally:
                        safe_remove(onnx_file)
                        safe_remove(onnx_file_inf)

                    per_result['oripeak'] = oripeak
                    total_latency = None

                    if len(which_over_mem)>0 and oripeak >= (vm_available*1024):
                        per_result['TS'] = True
                        peak_after_patch = oripeak
                        

                    else:
                        per_result['TS'] = False 
                        



                    if min_latency_result:
                        print("find min_latency_result")
                        peak_after_patch = min_latency_result['total_peak_mem']
                        per_result['ori_mac'] = min_latency_result['mac_ori']+min_latency_result['mac_other']
                        per_result['plus_mac'] = min_latency_result['mac_gap']
                        per_result['ori_access'] = min_latency_result['access_ori']+min_latency_result['access_other']
                        per_result['plus_access'] = min_latency_result['access_gap']
                        total_latency=0

                        # mac_count = min_latency_result['mac_ori']+min_latency_result['mac_gap']+min_latency_result['mac_other']
                        per_result['mem'] = peak_after_patch

                    elif min_mem_result:
                        print("find min_mem_result")
                        peak_after_patch = min_mem_result['total_peak_mem']
                        per_result['ori_mac'] = min_mem_result['mac_ori']+min_mem_result['mac_other']
                        per_result['plus_mac'] = min_mem_result['mac_gap']
                        per_result['ori_access'] = min_mem_result['access_ori']+min_mem_result['access_other']
                        per_result['plus_access'] = min_mem_result['access_gap']
                        total_latency=0

                        # mac_count = min_mem_result['mac_ori']+min_mem_result['mac_gap']+min_mem_result['mac_other']
                        per_result['mem'] = peak_after_patch

                    elif tried_but_not_valid:
                        print("didn't find valid patching")
                        peak_after_patch = tried_but_not_valid['total_peak_mem']
                        per_result['ori_mac'] = tried_but_not_valid['mac_ori']+tried_but_not_valid['mac_other']
                        per_result['plus_mac'] = tried_but_not_valid['mac_gap']
                        per_result['ori_access'] = tried_but_not_valid['access_ori']+tried_but_not_valid['access_other']
                        per_result['plus_access'] = tried_but_not_valid['access_gap']
                        per_result['mem'] = peak_after_patch
                        total_latency=0

                    else:
                        peak_after_patch = oripeak
                        per_result['mem'] = oripeak
                        total_latency = None
                        

                    if USE_LATENCY_PROXY and (total_latency==0):
                        total_latency = (per_result['ori_mac'] + per_result['plus_mac']) + LC_RATIO * (per_result['ori_access'] + per_result['plus_access'])
                        if total_latency > LC_PROXY:
                            print("Over latency constraint")
                            under_mem = False
                            per_result['under'] = False
                            check_per_results.append(per_result.copy())
                            continue

                    
                    if peak_after_patch >= (vm_available*1024):
                        print("after patchts, can not meet vm constraint")
                        under_mem = False
                        per_result['under'] = False
                        check_per_results.append(per_result.copy())
                        continue
                    else:
                        under_mem = True
                        per_result['under'] = True

                elif global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'nots':
                    try:
                        which_over_mem, oripeak, total_macs, total_access = model_tracing(onnx_path, model_name) 
                    finally:
                        safe_remove(onnx_file)
                        safe_remove(onnx_file_inf)

                    per_result['ori_mac'] = total_macs
                    per_result['plus_mac'] = 0
                    per_result['ori_access'] = total_access
                    per_result['plus_access'] = 0
                    per_result['mem'] = oripeak
                    print(f"which_over_mem: {which_over_mem}")
                    per_result['TS'] = False 

                    if which_over_mem == [] and oripeak <= (vm_available*1024):
                        under_mem = True
                        per_result['under'] = True
                
                    else:
                        under_mem = False
                        per_result['under'] = False


                    

                if not under_mem:
                    per_result['under'] = False
                    check_per_results.append(per_result.copy())
                    continue
            
            else: # no ts  
                per_result['under'] = True
                per_result['TS'] = False 

    
                #print("total_lan = ", total_lan, " under_mem = ", under_mem)
                #print("dup_path = ", dup_path, " est_peak_per_path = ",est_peak_per_path," q_list_per_path = ", q_list_per_path)

            

            subnet_latency_info = performance_model.get_latency_info(subnet_obj, subnet_cpb)
            if subnet_latency_info:
                   print("subnet_latency_info = ", subnet_latency_info['perf_e2e_contpow_fp_lat'])
            else:
                per_result['under'] = False
                check_per_results.append(per_result.copy())
                continue

            #print(subnet_latency_info)
            checked_constraints = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_CONSTRAINTS']
            
            ts_mode = global_settings.NAS_SETTINGS_GENERAL['MODE']


            _, network_nvm_usage, _ = performance_model.get_nvm_usage(subnet_obj)
            max_features = max((f+w for f, w in network_nvm_usage), default=0)
            per_result['nvm'] = max_features

            if not check_constraints(performance_model, subnet_latency_info, subnet_obj, subnet_name, subnet_cpb, checked_constraints, constraint_stats, ts_mode, under_mem):
                if under_mem:
                    per_result['under_nvm'] = False
                else:
                    per_result['under'] = False
                    per_result['under_nvm'] = False
                check_per_results.append(per_result.copy())
                continue
            else:
                per_result['under_nvm'] = True
                per_result['under'] = True
            
            # get perf for CONT pow                 
            # cont pow performance - best params
            network_flops_contpow, _, _ = performance_model.get_network_flops(subnet_obj,fixed_params=None, layer_based_cals=layer_based_cals)
            
            if network_flops_contpow:
                flops_sum = sum(network_flops_contpow)
            else:
                flops_sum = (per_result['ori_mac']+per_result['plus_mac']) *2

            if per_result['TS'] and not network_flops_contpow: # and constraint_stats['flops_max'] == float('-inf'):
                print("use mac_count")
                flops_sum =(per_result['ori_mac']+per_result['plus_mac']) *2
                constraint_stats['flops_max'] = max(flops_sum,constraint_stats['flops_max'])
                constraint_stats['flops_min'] = max(flops_sum,constraint_stats['flops_max'])
                per_result['flops'] = flops_sum

            else:
                flops_sum = sum(network_flops_contpow)
                constraint_stats['flops_max'] = max(sum(network_flops_contpow), constraint_stats['flops_max'])
                constraint_stats['flops_min'] = min(sum(network_flops_contpow), constraint_stats['flops_min'])
                per_result['flops'] = flops_sum

            
        except Exception as e:            
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc()
            print(tb)
            print("subnet_cpb: ", subnet_cpb)
            if (per_result['ori_mac']+per_result['plus_mac']) > 0:
                flops_sum = (per_result['ori_mac']+per_result['plus_mac']) *2
            else:
                flops_sum = 0
        
          
        if (error_net_perf == False):
            print("[CPU-{}] Finished processing subnet: {}, flops: under CONTpow={}".format(cpuid, subnet_name, network_flops_contpow))
        else:
            print("[CPU-{}] ERROR processing subnet: {}, flops: under CONTpow={}".format(cpuid, subnet_name, network_flops_contpow))
        
        #print("subnet_latency_info:", subnet_latency_info)

        if not subnet_latency_info or 'perf_e2e_contpow_fp_lat' not in subnet_latency_info:
            print(f"[CPU-{cpuid}] Skipping subnet {subnet_name} due to missing latency info")
            per_result['under'] = False
            check_per_results.append(per_result.copy())
            continue

        if flops_sum == 0:
            print(f"[CPU-{cpuid}] Skipping subnet {subnet_name} due to missing flops_sum")
            per_result['under'] = False
            check_per_results.append(per_result.copy())
            continue
        #if 'perf_e2e_contpow_fp_lat' not in subnet_latency_info:
        #    raise ValueError("subnet_latency_info is missing required key 'perf_e2e_contpow_fp_lat'")


        check_per_results.append(per_result.copy())
        print(f"[SUBCHECK]subnet: {per_result['id']}, oripeak:{per_result['oripeak']} , peak memory: {per_result['mem']}, flops: {per_result['flops']}, nvm: {per_result['nvm']}, under_const: {per_result['under']}, under_nvm: {per_result['under_nvm']}, ori_mac: {per_result['ori_mac']}, plus_mac: {per_result['plus_mac']}, ori_access: {per_result['ori_access']},plus_access: {per_result['plus_access']}, cpb: {per_result['cpb']}, TS: {per_result['TS']}\n")



        subnet_results.append({
            "subnet_name": subnet_name,
            "subnet_obj" : netobj_to_pyobj(subnet_obj),
            "subnet_choice_per_blk": subnet_cpb,
            "net_choices" : each_subnet.net_choices,

            "supernet_choice": (width_multiplier, input_resolution),

            # perf cont pow
            "perf_e2e_contpow_fp_lat": subnet_latency_info['perf_e2e_contpow_fp_lat'],
            "perf_e2e_contpow_flops": flops_sum,

        })

    return subnet_results, constraint_stats, check_per_results


def calc_flops_for_supernet(global_settings, dataset, supernet, width_multiplier, input_resolution, supernet_block_choices, n, layer_based_cals, first_block_hard_coded):

    # sampling in parent process, until there are n satisfying subnets
    all_subnet_configs_lst_generator = sample_subnet_configs(supernet, supernet_block_choices, first_block_hard_coded=first_block_hard_coded)

    available_cpus = min(16, get_max_num_workers(worker_type='CPU'))    # to ensure deterministic running, as all servers have at least 16 cores
    #available_cpus = max(16, get_max_num_workers(worker_type='CPU'))
    all_check_sp_results = []
    all_subnet_results = []

    while True:
        needs_subnets = n
        print(f'Sampling {needs_subnets} more subnet(s)')

        all_subnet_configs = list(itertools.islice(all_subnet_configs_lst_generator, needs_subnets))
        batched_subnet_configs = np.array_split(all_subnet_configs, available_cpus)

        results_all_cpus = run_multiprocessing_workers(
            num_workers=available_cpus,
            worker_func=flops_worker,
            worker_type='CPU',
            common_args=(global_settings, width_multiplier, input_resolution, layer_based_cals),
            worker_args=batched_subnet_configs,
        )
        all_constraint_stats = []
        for subnet_results_per_cpu, constraint_stats_per_cpu, check_sp_results_per_cpu in results_all_cpus:
            all_subnet_results.extend(subnet_results_per_cpu)
            all_constraint_stats.append(constraint_stats_per_cpu)
            all_check_sp_results.append(check_sp_results_per_cpu)

        constraint_stats = merge_constraint_stats(all_constraint_stats)
        print(constraint_stats)

        # break when there are enough subnets
        if len(all_subnet_results) >= n:
            break

        if not global_settings.NAS_SSOPTIMIZER_SETTINGS['DO_RESAMPLING']:
            break

    print("\n------ All jobs complete ------ for width_multiplier={}, input_resolution={}, time={}..\n\n".format(
        width_multiplier, input_resolution, datetime.datetime.now()))

    ts_mode = global_settings.NAS_SETTINGS_GENERAL['MODE']
    ts_goal = global_settings.NAS_SETTINGS_GENERAL['GOAL']
    txt_name = f"{ts_mode}_{ts_goal}_w{width_multiplier}_ir{input_resolution}_check_per_supernet.txt"

    # with open(txt_name, "w", encoding="utf-8") as file:
    #     for per_cpu_rst in all_check_sp_results:       # List for each CPU
    #         for per_rst in per_cpu_rst:               # Each result dict
    #             file.write(
    #                 f"subnet: {per_rst['id']}, "
    #                 f"ori_peak: {per_rst['oripeak']},"
    #                 f"peak memory: {per_rst['mem']}, "
    #                 f"flops: {per_rst['flops']}, "
    #                 f"nvm: {per_rst['nvm']}, "
    #                 f"under_const: {per_rst['under']}, "
    #                 f"under_nvm: {per_rst['under_nvm']}, "
    #                 f"ori_mac: {per_rst['ori_mac']},"
    #                 f"plus_mac: {per_rst['plus_mac']},"
    #                 f"ori_access: {per_rst['ori_access']},"
    #                 f"plus_access: {per_rst['plus_access']},"
    #                 f"cpb: {per_rst['cpb']}, "
    #                 f"TS: {per_rst['TS']}\n"
    #             )

    return {
        'all_subnet_results': all_subnet_results[:n],
        'constraint_stats': constraint_stats,
    }


def reorganize_flops_data(flops_data) -> Dict[NetChoice, List[Dict]]:
    all_subnets_with_flops = {}

    for subnet_data in flops_data:
        supernet_choice = tuple(subnet_data['supernet_choice'])
        all_subnets_with_flops.setdefault(supernet_choice, []).append((subnet_data['perf_e2e_contpow_flops'], subnet_data))

    return all_subnets_with_flops

def sort_by_average_flops(data):
    supernet_choice, cur_subnets_with_flops, average_flops = data
    return average_flops

def sorted_dict(d):
    # https://stackoverflow.com/questions/50493838/fastest-way-to-sort-a-python-3-7-dictionary
    return {k: d[k] for k in sorted(d)}

def ss_optimization_by_flops(global_settings, dataset, supernet_choices, supernet_block_choices):

    all_subnet_results = []
    per_supernet_stats = {}

    subnet_sample_size = global_settings.NAS_SSOPTIMIZER_SETTINGS['SUBNET_SAMPLE_SIZE']
    for width_multiplier, input_resolution in supernet_choices:
        supernet = get_supernet(global_settings=global_settings, dataset=dataset, width_multiplier=width_multiplier)
        ret = calc_flops_for_supernet(
            global_settings, dataset, supernet, width_multiplier, input_resolution, supernet_block_choices,
            n=subnet_sample_size,
            layer_based_cals=True, first_block_hard_coded=global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FIRST_BLOCK_HARD_CODED'])
        cur_subnet_results = ret['all_subnet_results']

        valid_subnets = len(cur_subnet_results)

        ret['constraint_stats']['num_subnets'] = valid_subnets
        if cur_subnet_results:
            ret['constraint_stats'].update({
                'latency_average': statistics.mean(subnet['perf_e2e_contpow_fp_lat'] for subnet in cur_subnet_results),
                'flops_average': statistics.mean(subnet['perf_e2e_contpow_flops'] for subnet in cur_subnet_results),
                })
        else:
            ret['constraint_stats'].update({
                'latency_average': None,
                'flops_average': None,
            })

        per_supernet_stats[f'({width_multiplier}, {input_resolution})'] = sorted_dict(ret['constraint_stats'])

        if valid_subnets >= global_settings.NAS_SSOPTIMIZER_SETTINGS['VALID_SUBNETS_THRESHOLD'] * subnet_sample_size:
            # drop the supernet if threshold is not reached
            all_subnet_results.extend(cur_subnet_results)
            print(f'There are {valid_subnets} valid subnets out of {subnet_sample_size} ones')
        else:
            print(f'Only {valid_subnets} valid subnets out of {subnet_sample_size} ones - dropping supernet')

    all_subnets_with_flops = reorganize_flops_data(all_subnet_results)

    supernet_average_flops = []

    # Put supernet choices and average flops for sorting
    for supernet_choice, cur_subnets_with_flops in all_subnets_with_flops.items():
        average_flops = statistics.mean(flops for flops, subnet_data in cur_subnets_with_flops)
        supernet_average_flops.append((
            supernet_choice,
            cur_subnets_with_flops,
            average_flops,
        ))
        print((supernet_choice, average_flops))

    
    supernet_average_flops.sort(key=sort_by_average_flops, reverse=True)    
    
    if len(supernet_average_flops) > 0:
        supernet_choice, cur_subnets_with_flops, average_flops = supernet_average_flops[0]
    else:
        supernet_choice = None; cur_subnets_with_flops = []; average_flops = None

    print('Stage 1 search space optimization done! Chosen supernet config: {}'.format(supernet_choice))

    supernet_properties = {
        'average_flops': average_flops,
        'num_subnets': len(cur_subnets_with_flops),
        'supernet_objtype': supernet.SUPERNET_OBJTYPE,
        'per_supernet_stats': per_supernet_stats,
    }

    return supernet_choice, supernet_properties
