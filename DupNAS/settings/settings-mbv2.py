import argparse
from enum import IntEnum, Enum
import json
import os, sys
from pathlib import Path
import pprint


from NASBase.model.shuffle_ss import (
    

    # IMAGE100
    SHUFFLE_STRIDE_FACTORS_IMAGE100,
    SHUFFLE_KERNEL_SIZES_IMAGE100,
    SHUFFLE_NUM_LAYERS_EXPLICIT_IMAGE100,

    SHUFFLE_WIDTH_MULTIPLIER_IMAGE100,
    SHUFFLE_INPUT_RESOLUTION_IMAGE100,

    SHUFFLE_NUM_OUT_CHANNELS_IMAGE100,

    # CIFAR 10
    SHUFFLE_STRIDE_FACTORS_CIFAR10,
    SHUFFLE_KERNEL_SIZES_CIFAR10,
    SHUFFLE_NUM_LAYERS_EXPLICIT_CIFAR10,

    SHUFFLE_WIDTH_MULTIPLIER_CIFAR10,
    SHUFFLE_INPUT_RESOLUTION_CIFAR10,

    SHUFFLE_NUM_OUT_CHANNELS_CIFAR10,
)

from NASBase.model.inception_ss import (
    
    # IMAGE100
    INCEPT_STRIDE_FACTORS_IMAGE100,
    INCEPT_KERNEL_SIZES_IMAGE100,
    INCEPT_MODULE_ABC_IMAGE100,
    INCEPT_NUM_LAYERS_EXPLICIT_IMAGE100,

    INCEPT_WIDTH_MULTIPLIER_IMAGE100,
    INCEPT_INPUT_RESOLUTION_IMAGE100,

    INCEPT_NUM_OUT_CHANNELS_IMAGE100,


    # CIFAR 10
    INCEPT_STRIDE_FACTORS_CIFAR10,
    INCEPT_KERNEL_SIZES_CIFAR10,
    INCEPT_MODULE_ABC_CIFAR10,
    INCEPT_NUM_LAYERS_EXPLICIT_CIFAR10,

    INCEPT_WIDTH_MULTIPLIER_CIFAR10,
    INCEPT_INPUT_RESOLUTION_CIFAR10,

    INCEPT_NUM_OUT_CHANNELS_CIFAR10,
)
from NASBase.model.mnas_ss import (
   
    # IMAGE100    
    EXP_FACTORS_IMAGE100,
    KERNEL_SIZES_IMAGE100,
    MOBILENET_NUM_LAYERS_EXPLICIT_IMAGE100,
    SUPPORT_SKIP_IMAGE100,
    
    WIDTH_MULTIPLIER_IMAGE100,
    INPUT_RESOLUTION_IMAGE100,
    
    MOBILENET_V2_NUM_OUT_CHANNELS_IMAGE100,

    # CIFAR 10
    EXP_FACTORS_CIFAR10,
    KERNEL_SIZES_CIFAR10,
    MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10,
    SUPPORT_SKIP_CIFAR10,

    WIDTH_MULTIPLIER_CIFAR10,
    INPUT_RESOLUTION_CIFAR10,

    MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10,

)

from NASBase.model.mbv2_ss import (
   
    # IMAGE100    
    EXP_FACTORS_IMAGE100,
    KERNEL_SIZES_IMAGE100,
    MOBILENET_NUM_LAYERS_EXPLICIT_IMAGE100,
    SUPPORT_SKIP_IMAGE100,
    
    WIDTH_MULTIPLIER_IMAGE100,
    INPUT_RESOLUTION_IMAGE100,
    
    MOBILENET_V2_NUM_OUT_CHANNELS_IMAGE100,

    # CIFAR 10
    EXP_FACTORS_CIFAR10,
    KERNEL_SIZES_CIFAR10,
    MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10,
    SUPPORT_SKIP_CIFAR10,

    WIDTH_MULTIPLIER_CIFAR10,
    INPUT_RESOLUTION_CIFAR10,

    MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10,

)




CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CURRENT_HOME_PATH = os.path.dirname(CURRENT_DIR_PATH)

class Stages(IntEnum):
    SS_OPT = 1
    TRAIN_SUPERNET = 2
    EVO_SEARCH = 3
    FINE_TUNE = 4

class SSOptPolicy(str, Enum):
    FLOPS = 'FLOPS'
    IMC = 'IMC'
    #OURS = 'OURS'

SETTINGS_CATEGORIES = (
    'GLOBAL_SETTINGS',
    'CUDA_SETTINGS',
    'PLATFORM_SETTINGS',
    'NAS_SETTINGS_GENERAL',
    'NAS_SSOPTIMIZER_SETTINGS',
    'NAS_EVOSEARCH_SETTINGS',
    'TINAS',
    'NAS_SETTINGS_PER_DATASET',
    'DUMPER_SETTINGS',
    'LOG_SETTINGS',
    'NAS_TESTING_SETTINGS',
)


class Settings(object): ##default settintgs & discription
    
    GLOBAL_SETTINGS = {
        # should be different for HAR/KWS
        'EXP_SUFFIX' : "test",
        'USE_REMOTE_LOGGER' : True,
        'REMOTE_LOGGER_RUN_NAME_SUFFIX': '',
        'REMOTE_LOGGER_GROUP_NAME_SUFFIX': '',
        'REMOTE_LOGGER_EXTRA_TAGS': [],
        'RLOGGER_PROJECT_NAME':'',

        # -------- Distributed flags (Stage-2 only) --------
        'DIST_MODE': 'none',      # [ADDED] none | ddp | fsdp
        'AMP': 'off',             # [ADDED] off | fp16 | bf16
        'BATCH_SCOPE': 'per_gpu', # [ADDED] per_gpu | global

    }    
    
    # ----------------------------------------------------
    # CUDA SETTINGS
    # ----------------------------------------------------
    CUDA_SETTINGS = {
        'GPUIDS' : "0,1,2,3", # GPU card
    }   
    # ----------------------------------------------------
    # PLATFORM SPECIFIC SETTINGS
    # ----------------------------------------------------
    PLATFORM_SETTINGS = {
        'MCU_TYPE' : 'MSP430',   # MSP430   |  MSP432
        'REHM' : 2800, # 75, 300.0,  # ehm equivalent resistance (ohm)
        'VSUP' : 5.892, # 5.0, 3.0 (V)
        'EAV_SAFE_MARGIN' : 0.60, # 0.10, 0.15, 0.20, ..., 0.55, # available energy will be reduced by this ratio
        'DATA_SZ' : 2, # data size in bytes  # check 16 or 8
        'POW_TYPE' : 'CONT',   #-> CONT
        'CPU_CLOCK': 16000000,         

        # Constraints:
        # * VM_CAPACITY for vm
        # * NVM_CAPACITY for nvm
        # * CCAP, VON and VOFF for energy per power cycle (cap_energy)
        # * LAT_E2E_REQ for latency
        # * IMC_CONSTRAINT for imc
        #

        'VM_CONSTRAINT': (128*1024),
        # A constraint is skipped if the value is <= 0        
        'VM_CAPACITY' : (512*1024),  # in bytes, note: leave room for application stack   #-> 256KB
        'NVM_CAPACITY' : (1000000),    # total capacity across one of more FRAM chips  #-> 1MB  #increase for debugging
        'NVM_CAPACITY_ALLOCATION' : [1000000, 1000000], # if two FRAM chips, one for features and another for weights #increase for debugging
        'LAT_E2E_REQ' : 10000, # by default no e2e latency constraint (seconds)
        'CCAP' : 0.005,  # capacitance (F)
        'VON' : 4.535, # (V)
        'VOFF' : 3.290, # (V)
        #'IMC_CONSTRAINT': 50,  # 50 means 50%
        
        # obtained by running tests on current EHM
        'REHM_TABLE':
            {
                "0.005" : 2800,
                "0.00047" : 3000,
                "0.0001":  3100
            }
    }



    # ----------------------------------------------------
    # NAS TOOL SETTINGS 
    # ----------------------------------------------------
    NAS_SETTINGS_GENERAL = {        
        'SEED'  : 123,    
        'MODEL_FN_TYPE' : 'MODEL_FN_CONV2D',      # [MODEL_FN_CONV2D | MODEL_FN_CONV1D | MODEL_FN_FC]
        'STAGES': '1,2,3,4',  # run all stages by default [1: ss_opt, 2: train_supernet, 3: evo_search, 4: fine_tune_best_sol]
               
        # related to training        
        'CHECKPOINT_DIR' : CURRENT_HOME_PATH + '/TiNAS/NASBase/checkpoints/', 
        'DATASET' : 'IMAGE100',
        
        # optimizer settings        
        'TRAIN_OPT_MOM' : 0.9,    # momentum
        'TRAIN_OPT_WD' : 5e-5, # 4e-5, #3e-4,    # weight decay

        'TRAIN_BATCHNORM_EPSILON': 1e-5,
        
        # debug related
        'TRAIN_PRINT_FREQ' : 100,   # print frequency of training      

        #NEW  "shuffle", "incept", "mbv2"  #no use: "mobile",
        'ARC': 'mbv2',
        'MODE': 'pdq',  #pdq,tinyts, tinynas
        'GOAL': 'bal',  #bal, mem
        'VMSIZE': 128,
        'EXP_FILE': False,

        'SEARCH_TIME_TESTING': False,

    }
    
    # Search space optimization default settings    
    NAS_SSOPTIMIZER_SETTINGS = {
        'SUBNET_SAMPLE_SIZE' : 200,    
        'VALID_SUBNETS_THRESHOLD': 0.1, # 0.05 or some other ratios
        'DO_RESAMPLING': False,
        'SSOPT_POLICY' : SSOptPolicy.FLOPS,
        # specify which constraints to consider
        # VM, NVM, ENERGY should be always checked
        'SSOPT_CONSTRAINTS': 'CHK_PASS_STORAGE,CHK_PASS_SPATIAL,CHK_PASS_ATOMICITY',   #-> REMOVE CHK_PASS_IMC',,CHK_PASS_RESPONSIVENESS
        'SSOPT_RESULTS_FNAME' : CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + GLOBAL_SETTINGS['EXP_SUFFIX'] + '_ssoptlog.json',
        'SSOPT_TRAINED_SUPERNET_FNAME' : CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + GLOBAL_SETTINGS['EXP_SUFFIX'] + '_trsupnetresults.json'        
    }
    
    # Evolutionary search default settings
    # POP_SIZE and GENERATIONS are per-dataset
    NAS_EVOSEARCH_SETTINGS = {
        'POP_SIZE' : 16,  # children num from PARENT_RATIO  #keep this size
        'GENERATIONS' : 15,  # total iterations num  #if need it, use large number for generations
        # use evo_hyperparam_tuning for PARENT_RATIO, MUT_PROB and MUT_RATIO
        'PARENT_RATIO' : 0.2,    #kept how many parents      
        'MUT_PROB': 0.05,    # probability
        'MUT_RATIO': 0.5,    #
        'EVOSEARCH_LOGFNAME' : CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + GLOBAL_SETTINGS['EXP_SUFFIX'] + "_evosearchlog.json",   
        'EVOSEARCH_SCORE_TYPE' : 'ACC',  # please see evolution_finder.py: ACC | ACC_IMC | ACC_IMO_LREQ | ACC_LREQ  #-> ACC
        'EVOSEARCH_TRIALS': 10,   # different seeds  # final test need to do more
        
        # Optional: This checks only NVM constraints, and it is only for getting results in a shorter time.        
        'EVOSEARCH_BYPASS_EFFICIENCY' : True,  #check this!!! #before: False

        # **For testing only**: keep sampled initial population in a file for reuse.
        # For testing different mutation and crossover strategies
        'EVOSEARCH_INITIAL_POPULATION_FNAME': None,
        
        'EVOSEARCH_ENABLE_EVOMEMORY' : False, # use caching mechanism   # work with or without table #before: False

        'FIXED_NUM_CPU_WORKERS': 4,
        
        'DEBUG_ENABLED' : False,
        'ONNX_FILE_PATH': CURRENT_HOME_PATH + "/TiNAS/NASBase/onnx_model/",
        'GEN_ALL_ONNX_FILE_PATH': CURRENT_HOME_PATH + "/genonnx/"+ NAS_SETTINGS_GENERAL['ARC']+'/',

        'LATENCY_RATIO' : 2,
        'LATENCY_PROXY' : 532837563 ,
        
        # LC based on analysis in proxy-time-regression 
        # [mbv2-vm128]     # LC_R1 = 532837563  LC_R2 = 786644670   LC_R3 = 1040451776
        # [shuffle-vm128]  # LC_R1 = 71042360  LC_R2 = 146685325   LC_R3 = 222328290
        # [incept-vm128]   # LC_R1 = 48519485  LC_R2 = 67437381   LC_R3 = 86355278

        
        # LC based on analysis in S3-min (without Nots)
        # [mbv2-vm128]     # LC_N/2 = 102338472  LC_N = 204676944
        # [shuffle-vm128]  # LC_N/2 = 49209136  LC_N = 98418272
        # [incept-vm128]   # LC_N/2 = 26247984  LC_N = 52495968

        # LC based on analysis in S3-min
        # [mbv2-vm128]     # LC_N = 89149488 LC_0.8N = 71319590 LC_0.5N = 44574744 
        # [shuffle-vm128]  # LC_N = 97828448 LC_0.8N = 78262758  LC_0.5N = 48914224 
        # [incept-vm128]   # LC_N = 34130700 LC_0.8N = 27304560  LC_0.5N = 17065350 
        
        # LC based on analysis in S1
        # [mbv2-vm128]     # LC_N = 717800000 LC_1.5N = 1076700000 LC_2N = 1435600000 LC_2.5N = 1794500000
        # [shuffle-vm128]  # LC_N = 218300000 LC_1.5N = 327450000  LC_2N = 436600000  LC_2.5N = 545750000
        # [incept-vm128]   # LC_N = 146200000 LC_1.5N = 219300000  LC_2N = 292400000  LC_2.5N = 365500000

        # LC based on NoneTS
        # [mbv2-vm128]     # 5N = 445747440   2N = 178298976  # N = 89149488
        # [shuffle-vm128]  # 5N = 611949600    2N = 244779840  # N = 122389920
        # [incept-vm128]   # 5N = 170653500    2N = 68261400  # N = 34130700
        
        # [shuffle-vm128] # LCours = 155906656  # LCtts = 261479072  # LCnas = 97828448
        # [mbv2-vm128]  # LCours = 1767840432 # LCtts = 652823472  # LCtnas = 204676944
        # [incept-vm128]  # LCours = 212169984  # LCtts = 33375456   # LCtnas = 54485136



             
    }



    # should also be per-dataset - override them in settings/xxx-har.json
    TINAS = {
        'STAGE1_SETTINGS': {},
        'STAGE2_SETTINGS': {
            # default: not dropping (the default NN search space for the specified dataset)
            # dropped: all blocks use the same dropped choices
            'BLOCK_SEARCH_SPACE': 'default',

            # For G1, G2
            # mutate_default: all blocks use the same probability
            # mutate_blockwise_prob: each block has its own mutation probability
            'MUTATION_OPERATOR': 'mutate_default',
            
            # For G2                        
            # Default
            'MUT_PROB_PER_BLOCK': [0.05, 0.05, 0.05, 0.05],
            # For exploitation, later during evo search, same prob as default
            'MUT_PROB_PER_BLOCK_EXPLOITATION': [0.05, 0.05, 0.05, 0.05],
            # For exploration, earlier during evo search, higher for first block (high IMO sensitivity)
            'MUT_PROB_PER_BLOCK_EXPLORATION': [0.2, 0.05, 0.05, 0.05],
            # After N generations, switching from exploration to exploitation
            'BEST_STABLE_GENERATIONS': 5,
        }
    }

    arc = NAS_SETTINGS_GENERAL['ARC']

    if arc == 'shuffle':
        TINAS['STAGE1_SETTINGS'] = {
            'DROPPING_BLOCK_LEVEL': {
                'STRIDE_FACTORS': [],
                'KERNEL_SIZES': [],
                'NUM_LAYERS_EXPLICIT': [],
            },
            'DROPPING_NET_LEVEL': {
                'WIDTH_MULTIPLIER': [],
                'INPUT_RESOLUTION': [],
            },
            'DROPPING_ENABLED': False,
        }

    elif arc == 'incept':
        TINAS['STAGE1_SETTINGS'] = {
            'DROPPING_BLOCK_LEVEL': {
                'STRIDE_FACTORS': [],
                'KERNEL_SIZES': [],
                'NUM_LAYERS_EXPLICIT': [],
                'MODULE_ABC': [],
            },
            'DROPPING_NET_LEVEL': {
                'WIDTH_MULTIPLIER': [],
                'INPUT_RESOLUTION': [],
            },
            'DROPPING_ENABLED': False,
        }

    elif arc == 'mobile':
        TINAS['STAGE1_SETTINGS'] = {
            'DROPPING_BLOCK_LEVEL': {
                'EXP_FACTORS': [],
                'KERNEL_SIZES': [],
                'NUM_LAYERS_EXPLICIT': [],
                'SUPPORT_SKIP': [],
            },
            'DROPPING_NET_LEVEL': {
                'WIDTH_MULTIPLIER': [],
                'INPUT_RESOLUTION': [],
            },
            'DROPPING_ENABLED': False,
        }
    elif arc == 'mbv2':
        TINAS['STAGE1_SETTINGS'] = {
            'DROPPING_BLOCK_LEVEL': {
                'EXP_FACTORS': [],
                'KERNEL_SIZES': [],
                'NUM_LAYERS_EXPLICIT': []
            },
            'DROPPING_NET_LEVEL': {
                'WIDTH_MULTIPLIER': [],
                'INPUT_RESOLUTION': [],
            },
            'DROPPING_ENABLED': False,
        }



    NAS_TESTING_SETTINGS = {
        'TRAINED_SUPERNET_FNAME': "",  # for independent testing
        # 'TRAINED_SUPERNET_SSOPT_LOGFNAME': "",        
        # Fine-tuning uses evo search result at the specified generation.
        'FINETUNE_BASE_GENERATION': None,
    }
    
    # ----------------------------------------------------
    # DATASET SPECIFIC SETTINGS 
    # ----------------------------------------------------
    
    NAS_SETTINGS_PER_DATASET = {}

    # Base settings for CIFAR10
    cifar10_base = {
        'NUM_BLOCKS': 3,
        'NUM_CLASSES': 10,
        'STEM_C_OUT': 16,
        'INPUT_CHANNELS': 3,
        'STRIDE_FIRST': 1,
        'DOWNSAMPLE_BLOCKS': [0, 1, 2, 3],
        'FIRST_BLOCK_HARD_CODED': False,
        'USE_1D_CONV': False,

        # Training-related
        'TRAIN_DATADIR': CURRENT_HOME_PATH + '/TiNAS/NASBase/dataset/CIFAR10',
        'TRAIN_OPT_LR': 0.025,
        'TRAIN_SUPERNET_BATCHSIZE': 16,
        'TRAIN_SUBNET_BATCHSIZE': 16,
        'VAL_BATCHSIZE': 16,
        'TRAIN_SUPERNET_EPOCHS': 75,
        'TRAIN_SUBNET_EPOCHS': 20,
        'FINETUNE_SUBNET_EPOCHS': 15,
        'FINETUNE_BATCHSIZE': 100,
        'FINETUNE_OPT_LR': 0.025,
    }

    # Base settings for IMAGE100
    image100_base = {
        'NUM_BLOCKS': 4,
        'NUM_CLASSES': 100,
        'STEM_C_OUT': 16,
        'INPUT_CHANNELS': 3,
        'STRIDE_FIRST': 1,
        'DOWNSAMPLE_BLOCKS': [0, 1, 2, 3],
        'FIRST_BLOCK_HARD_CODED': False,
        'USE_1D_CONV': False,

        # Training-related
        'TRAIN_DATADIR': CURRENT_HOME_PATH + '/TiNAS/NASBase/dataset/IMAGE100',
        'TRAIN_OPT_LR': 0.08,
        'TRAIN_SUPERNET_BATCHSIZE': 128,
        'TRAIN_SUBNET_BATCHSIZE': 128,
        'VAL_BATCHSIZE': 128,
        'TRAIN_SUPERNET_EPOCHS': 200,
        'TRAIN_SUBNET_EPOCHS': 25,
        'FINETUNE_SUBNET_EPOCHS': 200,
        'FINETUNE_BATCHSIZE': 64,
        'FINETUNE_OPT_LR': 0.2,
    }

    arc = NAS_SETTINGS_GENERAL['ARC']

    # Add architecture-specific settings
    if arc == 'shuffle':
        cifar10_base.update({
            'OUT_CH_PER_BLK': SHUFFLE_NUM_OUT_CHANNELS_CIFAR10,
            'STRIDE_FACTORS': SHUFFLE_STRIDE_FACTORS_CIFAR10,
            'KERNEL_SIZES': SHUFFLE_KERNEL_SIZES_CIFAR10,
            'NUM_LAYERS_EXPLICIT': SHUFFLE_NUM_LAYERS_EXPLICIT_CIFAR10,
            'WIDTH_MULTIPLIER': SHUFFLE_WIDTH_MULTIPLIER_CIFAR10,
            'INPUT_RESOLUTION': SHUFFLE_INPUT_RESOLUTION_CIFAR10,
        })
        image100_base.update({
            'OUT_CH_PER_BLK': SHUFFLE_NUM_OUT_CHANNELS_IMAGE100,
            'STRIDE_FACTORS': SHUFFLE_STRIDE_FACTORS_IMAGE100,
            'KERNEL_SIZES': SHUFFLE_KERNEL_SIZES_IMAGE100,
            'NUM_LAYERS_EXPLICIT': SHUFFLE_NUM_LAYERS_EXPLICIT_IMAGE100,
            'WIDTH_MULTIPLIER': SHUFFLE_WIDTH_MULTIPLIER_IMAGE100,
            'INPUT_RESOLUTION': SHUFFLE_INPUT_RESOLUTION_IMAGE100,
        })

    elif arc == 'incept':
        cifar10_base.update({
            'OUT_CH_PER_BLK': INCEPT_NUM_OUT_CHANNELS_CIFAR10,
            'STRIDE_FACTORS': INCEPT_STRIDE_FACTORS_CIFAR10,
            'KERNEL_SIZES': INCEPT_KERNEL_SIZES_CIFAR10,
            'MODULE_ABC': INCEPT_MODULE_ABC_CIFAR10,
            'NUM_LAYERS_EXPLICIT': INCEPT_NUM_LAYERS_EXPLICIT_CIFAR10,
            'WIDTH_MULTIPLIER': INCEPT_WIDTH_MULTIPLIER_CIFAR10,
            'INPUT_RESOLUTION': INCEPT_INPUT_RESOLUTION_CIFAR10,
        })
        image100_base.update({
            'OUT_CH_PER_BLK': INCEPT_NUM_OUT_CHANNELS_IMAGE100,
            'STRIDE_FACTORS': INCEPT_STRIDE_FACTORS_IMAGE100,
            'KERNEL_SIZES': INCEPT_KERNEL_SIZES_IMAGE100,
            'MODULE_ABC': INCEPT_MODULE_ABC_IMAGE100,
            'NUM_LAYERS_EXPLICIT': INCEPT_NUM_LAYERS_EXPLICIT_IMAGE100,
            'WIDTH_MULTIPLIER': INCEPT_WIDTH_MULTIPLIER_IMAGE100,
            'INPUT_RESOLUTION': INCEPT_INPUT_RESOLUTION_IMAGE100,
        })

    elif arc == 'mobile':
        cifar10_base.update({
            'OUT_CH_PER_BLK': MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10,
            'EXP_FACTORS': EXP_FACTORS_CIFAR10,
            'KERNEL_SIZES': KERNEL_SIZES_CIFAR10,
            'NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10,
            'SUPPORT_SKIP': SUPPORT_SKIP_CIFAR10,
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_CIFAR10,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_CIFAR10,
        })
        image100_base.update({
            'OUT_CH_PER_BLK': MOBILENET_V2_NUM_OUT_CHANNELS_IMAGE100,
            'EXP_FACTORS': EXP_FACTORS_IMAGE100,
            'KERNEL_SIZES': KERNEL_SIZES_IMAGE100,
            'NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_IMAGE100,
            'SUPPORT_SKIP': SUPPORT_SKIP_IMAGE100,
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_IMAGE100,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_IMAGE100,
        })
    elif arc == 'mbv2':
        cifar10_base.update({
            'OUT_CH_PER_BLK': MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10,
            'EXP_FACTORS': EXP_FACTORS_CIFAR10,
            'KERNEL_SIZES': KERNEL_SIZES_CIFAR10,
            'NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10,
            'SUPPORT_SKIP': SUPPORT_SKIP_CIFAR10,
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_CIFAR10,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_CIFAR10,
        })
        image100_base.update({
            'OUT_CH_PER_BLK': MOBILENET_V2_NUM_OUT_CHANNELS_IMAGE100,
            'EXP_FACTORS': EXP_FACTORS_IMAGE100,
            'KERNEL_SIZES': KERNEL_SIZES_IMAGE100,
            'NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_IMAGE100,
            'SUPPORT_SKIP': SUPPORT_SKIP_IMAGE100,
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_IMAGE100,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_IMAGE100,
        })

    # Assign the full dictionary
    NAS_SETTINGS_PER_DATASET['CIFAR10'] = cifar10_base
    NAS_SETTINGS_PER_DATASET['IMAGE100'] = image100_base
    
        
    
    # ----------------------------------------------------
    # DNN DUMPER SETTINGS
    # ----------------------------------------------------
    DUMPER_SETTINGS = {
        'DUMP_DIR' : '' #<where to store the solutions (*.h5 model, *.h model)
    }


    # ----------------------------------------------------
    # DNN DUMPER SETTINGS
    # ----------------------------------------------------
    LOG_SETTINGS = {    
        'TRAIN_LOG_DIR' : CURRENT_HOME_PATH + '/TiNAS/NASBase/train_log/', 
        'TRAIN_LOG_FNAME' : "train_info.csv", 
        'LOG_LEVEL' : 1, 
        'REMOTE_LOGGING_SYNC_DIR' : CURRENT_HOME_PATH + '/TiNAS/wandb_dir/'
    }

    def get_dict(self):
        result = {}
        for settings_category in SETTINGS_CATEGORIES:
            result[settings_category] = getattr(self, settings_category)
        return result
        
    
    def __init__(self):
        pass
    
    def __str__(self):
        result = ''
        for settings_category in SETTINGS_CATEGORIES:
            result += settings_category + ':=' + '\n'
            result += pprint.pformat(getattr(self, settings_category)) + "\n"
        return result

    # __getstate__ and __setstate__ needed to preserve settings across multiprocessing workers
    # https://docs.python.org/3.9/library/pickle.html#object.__getstate__

    def __getstate__(self):
        ret = {}
        for key, value in type(self).__dict__.items():
            if key.startswith('__'):
                continue
            ret[key] = value
        return ret

    def __setstate__(self, state):
        for key, value in state.items():
            d = type(self).__dict__[key]
            if isinstance(d, dict):
                d.update(value)


def load_settings(fname):
    # load json
    if os.path.exists(fname):
        json_data=open(fname)
        file_data = json.load(json_data)
        return file_data
    else:
        sys.exit("ERROR - file does not exist : " + fname)
        return None


def _update_settings(default_settings, new_settings):
    for k, v in new_settings.items():
        # adds new items, overwrites existing items
        if isinstance(default_settings[k], dict):
            # Update nested dictionaries recursively
            # Inspired by https://stackoverflow.com/a/3233356
            default_settings[k] = _update_settings(default_settings.get(k, {}), v)
        else:
            default_settings[k] = v
    return default_settings

def apply_settings_file(test_settings, settings_filenames):
    for settings_filename in settings_filenames.split(','):
        settings_json = load_settings(settings_filename)

        pprint.pprint(settings_json)

        apply_settings_json(test_settings, settings_json)

def apply_settings_json(test_settings, settings_json):
    for settings_category in SETTINGS_CATEGORIES:
        if settings_category in settings_json:
            old_settings = getattr(test_settings, settings_category)
            new_settings = _update_settings(old_settings, settings_json[settings_category])
            setattr(test_settings, settings_category, new_settings)


def arg_parser(test_settings):
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument('--gpuid',    type=str, default=argparse.SUPPRESS,  help="GPU selection")
    
    parser.add_argument('--dataset',  type=str,  default=argparse.SUPPRESS,  help="supported dataset including : 1. CIFAR10 (default), 2. IMAGE100")
    
    parser.add_argument('--ccap',    type=float, default=argparse.SUPPRESS,   help="capacitor size")
    parser.add_argument('--latreq',    type=float, default=argparse.SUPPRESS,   help="end-to-end latency requirement")
    parser.add_argument('--imcreq',    type=float, default=argparse.SUPPRESS,   help="end-to-end IMC requirement")
    parser.add_argument('--rehm',   type=float, default=argparse.SUPPRESS,   help="EHM equivalent resistance")

    parser.add_argument('--seed',    type=int, default=argparse.SUPPRESS,   help="seed for randomness, default is 123")
    parser.add_argument('--suffix',   type=str, default=argparse.SUPPRESS,   help="experiment run name suffix")
    parser.add_argument('--settings', type=str, default=argparse.SUPPRESS, help="settings files to load")
    parser.add_argument('--settings-json', type=str, default=argparse.SUPPRESS, help="settings data to load")
    parser.add_argument('--stages', type=str, default=argparse.SUPPRESS, help="stages to run, as comma-separated integers : " + ', '.join('{}. {}'.format(p.value, p.name) for p in Stages))
    parser.add_argument('--ss-opt-policy', type=str, default=argparse.SUPPRESS, help='search space optimization policy : ' + ', '.join(p.value for p in SSOptPolicy))
    parser.add_argument('--tr-sup-fname', type=str, default=argparse.SUPPRESS, help="the filename of the trained supernet") # for independent testing
    parser.add_argument('--tr-sup-config', type=str, default=argparse.SUPPRESS, help="the config of the trained supernet") # for independent testing

    parser.add_argument('--no-rlogger', action="store_true", default=False,  help="switch off remote logger")
    parser.add_argument('--rlogger-proj-name', type=str, default=argparse.SUPPRESS, help='Project name for the remote logger')

    #----new for TS
    parser.add_argument("--arc", type=str, default="mobile", choices=["mobile", "shuffle", "incept", "mbv2"],  #none: no any TS
                    help="Choose one of: mobile, shuffle, incept")
    parser.add_argument("--mode", type=str, default="none", choices=["pdq", "tinyts", "tinynas", "none"],  #none: no any TS
                    help="Choose one of: pdq, tinyts, tinynas, none (none means no any TS)")
    parser.add_argument("--priority", type=str, default="bal", choices=["bal", "mem"],  #none: no any TS
                    help="Choose one of: pdq, bal, mem (goal)")
    parser.add_argument("--export_file", action='store_true',
                    help="Enable exporting reports and figures")
    parser.add_argument("--vmsize", type=int, default=512,
                    help="Set memory constraint in KB (e.g., 32 for 32KB)")
    
    # [ADDED] Stage-2 distributed knobs (used only in TRAIN_SUPERNET)
    parser.add_argument("--dist", choices=["none","ddp","fsdp"], default="none",
                    help="Use multi-GPU ONLY for Stage 2 supernet training")  # [ADDED]
    parser.add_argument("--amp", choices=["off","fp16","bf16"], default="off",
                    help="Mixed precision for Stage 2 (ddp/fsdp)")  # [ADDED]
    parser.add_argument("--batch-scope", choices=["per_gpu","global"], default="per_gpu",
                    help="Interpret TRAIN_*_BATCHSIZE as per-GPU or global in Stage 2")  # [ADDED]


    args = parser.parse_args()

    if 'arc' in args:
        print('ARG_SET_ARCHITECTURE : ', args.arc)
        test_settings.NAS_SETTINGS_GENERAL['ARC'] = args.arc
    if 'mode' in args:
        print('ARG_SET_MODE : ', args.mode)
        test_settings.NAS_SETTINGS_GENERAL['MODE'] = args.mode
    if 'priority' in args:
        print('ARG_SET_MODE : ', args.mode)
        test_settings.NAS_SETTINGS_GENERAL['GOAL'] = args.priority
    if 'vmsize' in args:
        print('ARG_SET_VM_SIZE : ', args.vmsize)
        test_settings.NAS_SETTINGS_GENERAL['VMSIZE'] = args.vmsize
        test_settings.PLATFORM_SETTINGS['VM_CONSTRAINT'] = int(args.vmsize)*1024
    if 'export_file' in args:
        print('ARG_SET: PRINT DETAILED FILES')
        test_settings.NAS_SETTINGS_GENERAL['EXP_FILE'] = args.export_file

    # first apply settings file
    if 'settings' in args:
        print('ARG_IMPORT_SETTINGS : %s'%(args.settings))
        apply_settings_file(test_settings, args.settings)
    if 'settings_json' in args:
        print('ARG_IMPORT_SETTINGS_JSON : %r' % (args.settings_json))
        apply_settings_json(test_settings, json.loads(args.settings_json))
        
    # then apply custom fine-grain settings
    if 'gpuid' in args:
        print('ARG_SET_GPUIDS : ', args.gpuid)
        test_settings.CUDA_SETTINGS['GPUIDS'] = args.gpuid
    if 'dataset' in args:
        print('ARG_SET_DATASET : ', args.dataset)
        test_settings.NAS_SETTINGS_GENERAL['DATASET'] = args.dataset
    
    if 'seed' in args:
        print('ARG_SET_SEED : ', args.seed)
        test_settings.NAS_SETTINGS_GENERAL['SEED'] = args.seed
    if 'suffix' in args:
        print('ARG_SET_SUFFIX : ', args.suffix)
        test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] = args.suffix

    if 'ccap' in args:
        print('ARG_SET_CCAP : ', args.ccap)
        test_settings.PLATFORM_SETTINGS['CCAP'] = args.ccap
    if 'latreq' in args:
        print('ARG_SET_LATREQ : ', args.latreq)
        test_settings.PLATFORM_SETTINGS['LAT_E2E_REQ'] = args.latreq
    if 'imcreq' in args:
        print('ARG_SET_LATREQ : ', args.imcreq)
        test_settings.PLATFORM_SETTINGS['IMC_CONSTRAINT'] = args.imcreq
    if 'rehm' in args:
        print('ARG_SET_REHM : ', args.rehm)
        test_settings.PLATFORM_SETTINGS['REHM'] = args.rehm
    if 'stages' in args:
        print('ARG_STAGES : ', args.stages)
        test_settings.NAS_SETTINGS_GENERAL['STAGES'] = args.stages
    if 'tr_sup_fname' in args:
        print('ARG_TRAINED_SUPERNET_FNAME : ', args.tr_sup_fname)
        test_settings.NAS_TESTING_SETTINGS['TRAINED_SUPERNET_FNAME'] = args.tr_sup_fname
    if 'tr-sup-config' in args:
        print('ARG_TRAINED_SUPERNET_CONFIG : ', args.tr_sup_config)
        test_settings.NAS_TESTING_SETTINGS['TRAINED_SUPERNET_CONFIG'] = args.tr_sup_config

    print('ARG_NO_RLOGGER : ', args.no_rlogger)
    test_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER'] = not args.no_rlogger
    if 'rlogger_proj_name' in args:
        print('ARG_RLOGGER_PROJ_NAME : ', args.rlogger_proj_name)
        test_settings.GLOBAL_SETTINGS['RLOGGER_PROJECT_NAME'] = args.rlogger_proj_name
    
    # [ADDED] stash Stage-2 dist knobs in settings
    test_settings.GLOBAL_SETTINGS['DIST_MODE'] = args.dist
    test_settings.GLOBAL_SETTINGS['AMP'] = args.amp
    test_settings.GLOBAL_SETTINGS['BATCH_SCOPE'] = args.batch_scope

    test_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + '_ssoptlog.json'
    test_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_TRAINED_SUPERNET_FNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + '_trsupnetresults.json'
    test_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + "_evosearchlog.json"

    if 'rehm' not in args:
        cap_str = str(test_settings.PLATFORM_SETTINGS['CCAP'])
        estimated_rehm = test_settings.PLATFORM_SETTINGS['REHM_TABLE'][cap_str]
        test_settings.PLATFORM_SETTINGS['REHM'] = estimated_rehm

    print('Updated settings:')
    print(str(test_settings))





    return test_settings

