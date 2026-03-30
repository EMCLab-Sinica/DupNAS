'''
Randomly sample subnets from supernets
'''
import socket
import argparse
import logging
import os, sys
import time
from datetime import datetime
import math
from pprint import pprint
import itertools
import numpy as np
import copy
import ast
import traceback

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchinfo import summary
#import multiprocessing
import torch.multiprocessing as mp



sys.path.append("..")

#sys.path.append(".")
from NASBase import utils as utils
from NASBase import file_utils as file_utils

#sys.path.append("..")
from settings import Settings, arg_parser, load_settings



if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mbv2':
    from NASBase.model.mbv2_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.mbv2_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    from NASBase.model.shuffle_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.shuffle_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    from NASBase.model.inception_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.inception_ss import *



#from NASBase.model.common_utils import *
from NASBase.hw_cost.Modules_nas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.hw_cost.Modules_nas_v1.CostModel.cnn import pass_constraint_storage

from NASBase.train_supernet import get_dataset

from NASBase.model.common_utils import (
    blkchoices_ixs_to_blkchoices, blkchoices_to_blkchoices_ixs, get_network_dimension, get_network_obj, get_subnet, get_supernet, iter_blk_choices, netobj_to_pyobj, 
    get_sampled_subnet_configs, get_subnet_from_config,
    get_dummy_net_input_tensor, netobj_to_pyobj
)

from NASBase import multiprocessing_helper as mp_helper

from logger.remote_logger import RemoteLogger


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


SUPERNET_TRAIN_CHKPNT_FNAME = "<TO_ADD>" # e.g., DupNAS-M-threshold-0.1_supernet_mnas_best.pth
BATCHED_SUBNET_RESULTS_FNAME_PREFIX = "<TO_ADD>" # e.g., load_supernet_batched_subnet_results_
COMBINED_SUBNET_RESULTS_FNAME = "<TO_ADD>" #e.g., "load_supernet_combined_results_"

GET_SUBNET_ACCURACY = False
GET_SUBNET_LATENCY = True
#GET_SUBNET_FLOPS = False

#DATASET = 'CIFAR10'

AVAILABLE_GPUIDS = [0,1,2,3]
NUM_CPU_CORES = 32
# AVAILABLE_CPUIDS = np.arange(NUM_CPU_CORES)

NUM_SUBNET_SAMPLES = 1000
NUM_SUBNET_SAMPLED_SUBBATCHSZ = 250
ENABLE_NVM_CONSTRAINT = True
OVERRIDE_NVM_SIZE = None

WIDTH_MULTIPLIER = 0.4
INPUT_RESOLUTION = 32

LOG_SUBDIR = "supernet_randsample/"

###########################################
# EXT SETTINGS CLASSES
###########################################




###########################################
# HELPERS
###########################################
# def _get_subnet_results_fname(global_settings, cpuid, res_type):
#     fname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + TRAIN_LOG_SUBDIR + BATCHED_SUBNET_RESULTS_FNAME_PREFIX + res_type + "_" + str(cpuid)+".json"
#     return fname

def _get_remote_logger_init_params(exp_type, server, global_settings, exp_consts, run_name_suffix="", group_name_suffix=""):
    
    init_params = {
        "rlog_proj_name" : "supernet_randsample",
        "rlog_run_name" : exp_type+run_name_suffix,
        "rlog_run_group" : exp_type+group_name_suffix,
        "rlog_run_config" : {            
                "script_name" : "supernet_randsample.py",
                "server"   : server,         
                "exp_start" : datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "settings_obj" : global_settings.get_dict(),
                "exp_constants" : exp_consts
            },
        "rlog_run_tags" : [exp_type 
                           ]
    }
    
    return init_params
    
    
def _get_remote_logger_obj(exp_type, global_settings, exp_consts, run_name_suffix="", group_name_suffix=""):
    
    servername = socket.gethostname()
    
    rl_init_params =  _get_remote_logger_init_params(exp_type, servername, global_settings, exp_consts, 
                                                     run_name_suffix=run_name_suffix,
                                                     group_name_suffix=group_name_suffix)
    RLogger = RemoteLogger(
                            proj_name = rl_init_params['rlog_proj_name'],                            
                            run_name = rl_init_params['rlog_run_name'],
                            config = rl_init_params['rlog_run_config'], 
                            tags=rl_init_params["rlog_run_tags"],
                            group=rl_init_params["rlog_run_group"]
                            )
    RLogger.init()
    return RLogger
        



# def _update_rehm(global_settings: Settings):
#     cap_str = 0.005
#     estimated_rehm = global_settings.PLATFORM_SETTINGS['REHM_TABLE'][cap_str]
#     global_settings.PLATFORM_SETTINGS['REHM'] = estimated_rehm
#     return global_settings

def _change_constraints_settings(sett, nvm_size):
    #large_nvm_capacity = 200000000
    sett.PLATFORM_SETTINGS['NVM_CAPACITY'] = nvm_size
    sett.PLATFORM_SETTINGS['NVM_CAPACITY_ALLOCATION'] = [int(nvm_size/2), int(nvm_size/2)]

def _update_dataset_specific_settings(dataset, global_settings: Settings):
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']    
    # default is cifar, settings are okay
    if dataset == 'CIFAR10':
        pass    
        


# remove nvm capacity 
def remove_subnets_exceed_nvm(global_settings, subnet_batch, supernet_blk_choices, 
                              width_multiplier, input_resolution, dataset):
    
    if ENABLE_NVM_CONSTRAINT == False:
        large_sz = 200000000
        _change_constraints_settings(global_settings, large_sz)
    
    filtered_batch = []
    for six, subnet_choice_per_blk_ixs in enumerate(subnet_batch):
        
        # -- get subnet obj
        #each_subnet_config = each_subnet_config.tolist()       
        #subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, each_subnet_config)   
        
        subnet_name = "SN-NET{}_{}-{}".format(width_multiplier, input_resolution, six)
             
        subnet_pyt = get_subnet(global_settings, dataset, supernet_blk_choices, subnet_choice_per_blk_ixs, six, 
                                width_multiplier=width_multiplier, input_resolution=input_resolution,
                                subnet_name=subnet_name)
                
        #input_tensor = torch.rand(1, 3, input_resolution, input_resolution)
        input_tensor = get_dummy_net_input_tensor(global_settings, input_resolution)        
        subnet_dims = get_network_dimension(subnet_pyt, input_tensor)                 
        subnet_obj = get_network_obj(subnet_dims)             
                
        all_layers_fit_nvm, _, network_nvm_usage = pass_constraint_storage(subnet_obj, global_settings.PLATFORM_SETTINGS)
    
        if all_layers_fit_nvm:
            filtered_batch.append(subnet_choice_per_blk_ixs)
            
        print("filtered_batch size = {}".format(len(filtered_batch)), end='\r')
        
    return filtered_batch
            
    
def remove_duplicates(subnet_batch_ixs, supernet_blk_choices):
    # esier to work with indeces
    # sets can't work with 2d lists, to convert to str
    subnet_batch_str = [str(s) for s in subnet_batch_ixs]        
    filtered_batch_ixs = [ast.literal_eval(s) for s in set(subnet_batch_str)]  
    #filtered_batch = [blkchoices_ixs_to_blkchoices(s, supernet_blk_choices) for s in filtered_batch_ixs]        
    return filtered_batch_ixs



# -- untested ---
def mpworker_subnets_accuracy(worker_id, global_settings, supernet_ckpt_fname, supernet_blk_choices, width_multiplier, input_resolution, subnet_batch):
    
    # --- common initialize
    device = torch.device("cuda:"+str(worker_id) if torch.cuda.is_available() else "cpu")                
    print (device)    
    torch.set_num_threads(1)
    
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']    
    
    # load model
    model = get_supernet(global_settings, dataset, 
                         load_state= True, supernet_train_chkpnt_fname=supernet_ckpt_fname,
                         width_multiplier=width_multiplier, input_resolution=input_resolution, 
                         blk_choices=supernet_blk_choices)
    
    # init cuda
    criterion = nn.CrossEntropyLoss().to(device)    
    model = model.to(device)    
    model.eval()
    
    # get dataset
    train_loader, val_loader = get_dataset(global_settings, num_workers=0)
        
    all_subnet_accuracy = []
    for six, sn_cpb in enumerate(subnet_batch):            
        # -- look up accuracy from supernet
        if not isinstance(sn_cpb, list):            
            sn_cpb = sn_cpb.tolist() # sample may be a list or a numpy array, while blkchoices_to_blkchoices_ixs works with lists only
        subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, sn_cpb)       
        
        # get accuracy   
        val_loss = utils.AverageMeter(); val_acc = utils.AverageMeter()        
        val_loss.reset(); val_acc.reset()
        
        with torch.no_grad(): # inference only
            for step, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, subnet_choice_per_blk_ixs)
                loss = criterion(outputs, targets)
                prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                n = inputs.size(0)
                val_loss.update(loss.item(), n)
                val_acc.update(prec1.item(), n)  
        
        all_subnet_accuracy.append({
                                        'sn_cpb' : sn_cpb,
                                        'val_acc' : val_acc.avg,
                                        'val_loss' : val_loss.avg
                                    }) 
        
        print("mpworker_subnets_accuracy:: [wid={}] <{},{}> processing batched subnets, progress={}/{}, val_acc={}".format(worker_id, 
                                                                                                            width_multiplier, input_resolution,
                                                                                                            six, len(subnet_batch), 
                                                                                                            val_acc.avg
                                                                                                            ))    
    
    return all_subnet_accuracy       
                
                
        
class Settingsplat(Settings):
    def __init__(self):
        self.PLATFORM_SETTINGS = copy.deepcopy(Settings.PLATFORM_SETTINGS)


def mpworker_subnets_efficiency(worker_id, platform_settings, exp_suffix, dataset, supernet_blk_choices, width_multiplier, input_resolution, subnet_batch):
    supernet_config = (width_multiplier, input_resolution)

    global_settings_plat = Settingsplat()

    for k, v in platform_settings.items():
        global_settings_plat.PLATFORM_SETTINGS[k] = v    

    global_settings_plat.NAS_SETTINGS_GENERAL['DATASET'] = dataset

    exp_consts = {
        "ENABLE_NVM_CONSTRAINT" : ENABLE_NVM_CONSTRAINT
    }
    
    rlog = _get_remote_logger_obj("load_supernet", global_settings_plat, exp_consts, 
                                  run_name_suffix="_wid{}".format(worker_id),
                                  group_name_suffix="_{}_<wm={},ir={}>".format(exp_suffix, width_multiplier, input_resolution))
    
    # -- get latency
    if ENABLE_NVM_CONSTRAINT == False:
        _change_constraints_settings(global_settings_plat)  
    
    
    performance_model_plat = PlatPerf(global_settings_plat.NAS_SETTINGS_GENERAL, global_settings_plat.PLATFORM_SETTINGS)
    
    #print(type(subnet_batch), np.shape(subnet_batch), len(subnet_batch)); sys.exit()
    
    batched_results = []
    for six, sn_cpb in enumerate(subnet_batch):  
        sn_cpb = sn_cpb.tolist()  
             
        subnet_obj, subnet_pyt = get_subnet_from_config(global_settings_plat, dataset, sn_cpb, supernet_config, subnet_idx=six)
            
        subnet_latency_info = PlatPerf.get_inference_latency_verbose(global_settings_plat, performance_model_plat, subnet_obj, sn_cpb)
        
        subnet_latency_info['subnet_obj'] = "Omitted for brevity" # removed to reduce json filesize
        
        # -- get flops
        subnet_flops, _ , flops_error = global_settings_plat.get_network_flops(subnet_obj, fixed_params=None, layer_based_cals=True)
        if (subnet_latency_info != None):
            subnet_latency_info['flops'] = np.sum(subnet_flops)
        
        #pprint(subnet_result)
        
        #if (subnet_latency_info['error_codes'] == [False, None, None]):        
        batched_results.append(subnet_latency_info)    
        
        rlog.log({"wid" : worker_id,
                "wm_ir" : "<{}_{}>".format(width_multiplier, input_resolution), 
                "progress" : round(((six+1)/len(subnet_batch))*100,2),                    
                "sn_cpb" : sn_cpb,
                "flops" : subnet_latency_info['flops'],
                "lat" : batched_results[-1]['perf_e2e_contpow_fp_lat']        
                })
            
             
    return batched_results




def run_multiple_supernets_latency(global_settings: Settings):
    
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']    
    settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[dataset]
    
    pprint(settings_per_dataset)
        
    net_search_options = {
        'WIDTH_MULTIPLIER' : settings_per_dataset['WIDTH_MULTIPLIER'],
        'INPUT_RESOLUTION' : settings_per_dataset['INPUT_RESOLUTION'],                
    }
        
    exp_suffix = global_settings.GLOBAL_SETTINGS['EXP_SUFFIX']
    date_str = datetime.today().strftime('%m%d%Y')
    train_log_subdir = LOG_SUBDIR+"load_supernet_NVM1MB_{}_{}/".format(exp_suffix, date_str)

    file_utils.dir_create(global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + train_log_subdir)
    
    supernet_blk_choices = iter_blk_choices(settings_per_dataset['STRIDE_FACTORS'], settings_per_dataset['KERNEL_SIZES'])

    
    platform_settings = global_settings.PLATFORM_SETTINGS   
     
    
    for each_wm in net_search_options['WIDTH_MULTIPLIER']:
        for each_ir in net_search_options['INPUT_RESOLUTION']:            
            print("==== run_multiple_supernets_latency:: running test: [{},{}] ======".format(each_wm, each_ir))
            
            sampled_subnet_configs_ixs = []
            resampling=0
            while (len(sampled_subnet_configs_ixs)<NUM_SUBNET_SAMPLES):
                        
                tmp_sampled_subnet_configs = get_sampled_subnet_configs(global_settings, dataset, supernet_blk_choices, n_rnd_samples=NUM_SUBNET_SAMPLED_SUBBATCHSZ)
                tmp_sampled_subnet_configs_ixs = [blkchoices_to_blkchoices_ixs(supernet_blk_choices, s)  for s in tmp_sampled_subnet_configs]        
                
                if ENABLE_NVM_CONSTRAINT == True:
                    tmp_sampled_subnet_configs_ixs = remove_subnets_exceed_nvm(global_settings, tmp_sampled_subnet_configs_ixs, supernet_blk_choices, 
                                                                               each_wm, each_ir, dataset)        
                
                sampled_subnet_configs_ixs.extend(tmp_sampled_subnet_configs_ixs)                                
                sampled_subnet_configs_ixs = remove_duplicates(sampled_subnet_configs_ixs, supernet_blk_choices)
                
                print("num valid unique subnets = ", len(sampled_subnet_configs_ixs))    
                resampling+=1
                
                if resampling>100: # tried to resample many times but still cannot find enough unique subnets
                    print("------- WARNING: cannot find sufficient number of unique valid subnets - [supnet={},{}][sz={}]".format(each_wm, each_ir, len(sampled_subnet_configs_ixs)))
                    break
                                        
            sampled_subnet_configs =  [blkchoices_ixs_to_blkchoices(s, supernet_blk_choices)  for s in sampled_subnet_configs_ixs]    
            
            
            # =============== GET LATENCY =============                    
            lat_results = []    
            #available_cpus = mp_helper.get_max_num_workers(worker_type='CPU')
            available_cpus = NUM_CPU_CORES
            batched_subnets = np.array_split(sampled_subnet_configs, available_cpus)
            
            all_worker_results = mp_helper.run_multiprocessing_workers(
                num_workers=available_cpus,
                worker_func= mpworker_subnets_efficiency,
                worker_type='CPU',
                common_args=(platform_settings, exp_suffix, dataset, supernet_blk_choices, each_wm, each_ir),
                worker_args=(batched_subnets),
            )		
            # -- combine results        
            for worker_result in all_worker_results:
                lat_results.extend(worker_result)    
                
            combined_results = {
                "lat_results" : lat_results,                
            }
            
            logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + train_log_subdir + "result_load_supernet_{}_{}_{}.json".format(dataset, each_wm, each_ir)
            file_utils.delete_file(logfname)   
            file_utils.json_dump(logfname, combined_results)    
                
                
                
    

def run(global_settings: Settings):
    
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']    
    settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[dataset]    
    
    # get all subnet choice per blks
    #model = get_supernet(global_settings, DATASET, load_state=False)
    supernet_ckpt_fname = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR'] + SUPERNET_TRAIN_CHKPNT_FNAME
        
    #supernet_blk_choices = model.blk_choices    
    supernet_blk_choices = iter_blk_choices(settings_per_dataset['STRIDE_FACTORS'], settings_per_dataset['KERNEL_SIZES'])
    #                                        settings_per_dataset['EXP_FACTORS'], 
    #                                        settings_per_dataset['KERNEL_SIZES'], 
    #                                        settings_per_dataset['MOBILENET_NUM_LAYERS_EXPLICIT'], 
    #                                        settings_per_dataset['SUPPORT_SKIP'])
    
    (width_multiplier, input_resolution) = (settings_per_dataset['WIDTH_MULTIPLIER'], settings_per_dataset['INPUT_RESOLUTION'])
    
    # -- sample subnets from trained supernet        
    #sampled_subnet_configs = get_sampled_subnet_configs(global_settings, DATASET, supernet_blk_choices, n_rnd_samples=NUM_SUBNET_SAMPLES)
        
    sampled_subnet_configs_ixs = []
    while (len(sampled_subnet_configs_ixs)<NUM_SUBNET_SAMPLES):
                
        tmp_sampled_subnet_configs = get_sampled_subnet_configs(global_settings, dataset, supernet_blk_choices, n_rnd_samples=NUM_SUBNET_SAMPLED_SUBBATCHSZ)
        tmp_sampled_subnet_configs_ixs = [blkchoices_to_blkchoices_ixs(supernet_blk_choices, s)  for s in tmp_sampled_subnet_configs]        
        tmp_sampled_subnet_configs_ixs = remove_subnets_exceed_nvm(global_settings, tmp_sampled_subnet_configs_ixs, supernet_blk_choices, 
                                                                   width_multiplier, input_resolution, dataset)        
        sampled_subnet_configs_ixs.extend(tmp_sampled_subnet_configs_ixs)
        
        sampled_subnet_configs_ixs = remove_duplicates(sampled_subnet_configs_ixs, supernet_blk_choices)
        
        print("num valid unique subnets = ", len(sampled_subnet_configs_ixs))   
    
    print("num valid unique subnets = ", len(sampled_subnet_configs_ixs))      
            
    sampled_subnet_configs =  [blkchoices_ixs_to_blkchoices(s, supernet_blk_choices)  for s in sampled_subnet_configs_ixs]    
    
    lat_results = []    
    acc_results = []
    
    # =============== GET LATENCY =============        
    if (GET_SUBNET_LATENCY):
        available_cpus = mp_helper.get_max_num_workers(worker_type='CPU')
        batched_subnets = np.array_split(sampled_subnet_configs, available_cpus)
        
        all_worker_results = mp_helper.run_multiprocessing_workers(
            num_workers=available_cpus,
            worker_func= mpworker_subnets_efficiency,
            worker_type='CPU',
            common_args=(global_settings, dataset, supernet_blk_choices, width_multiplier, input_resolution),
            worker_args=(batched_subnets),
        )		
        # -- combine results        
        for worker_result in all_worker_results:
            lat_results.extend(worker_result)
    
    
    # =============== GET ACCURACY =============    
    if (GET_SUBNET_ACCURACY):       
        available_gpus = mp_helper.get_max_num_workers('GPU')			
        batched_subnets = np.array_split(sampled_subnet_configs, available_gpus)
        all_worker_results = mp_helper.run_multiprocessing_workers(
            num_workers=available_gpus,
            worker_func= mpworker_subnets_accuracy,
            worker_type='GPU',
            common_args=(global_settings, supernet_ckpt_fname, supernet_blk_choices, width_multiplier, input_resolution), 
            worker_args=(batched_subnets),					
        )
        
        # -- combine results
        for worker_result in all_worker_results:
            acc_results.extend(worker_result)
        
    combined_results = {
        "lat_results" : lat_results,
        "acc_results" : acc_results,
    }
    
    logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + os.path.basename(supernet_ckpt_fname) + "_result_load_supernet.json"
    file_utils.delete_file(logfname)   
    file_utils.json_dump(logfname, combined_results)
    
    





if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #'0,1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '' #'0,1'
    
    test_settings = Settings() # default settings
    test_settings = arg_parser(test_settings)
    #run(test_settings)
    
    #test_settings = _update_rehm(test_settings)
        
    run_multiple_supernets_latency(test_settings)
    
        
    

