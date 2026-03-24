import os,copy
import random
from tqdm import tqdm
import numpy as np
import os.path
import sys
import itertools
from pprint import pprint
import multiprocessing

from datetime import datetime
import time
import torch
import onnx

# timestamp = time.time()
# date_time = datetime.fromtimestamp(timestamp)
# str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
# #print("Current timestamp", str_date_time)

#sys.path.append("../..")
from logger.remote_logger import get_remote_logger_obj
from settings import Settings

#sys.path.append("..")
#from NASBase.model.mnas_arch import MNASSuperNet
from NASBase.model.common_utils import (split_list_chunks, drop_choices, get_subnet_from_config,
    blkchoices_to_blkchoices_ixs, get_subnet, parametric_supernet_blk_choices,
    get_dummy_net_input_tensor, get_network_dimension, get_network_obj,
    )

if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mobile':
    from NASBase.model.mnas_arch import MNASSuperNet
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    from NASBase.model.shuffle_arch import MNASSuperNet
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    from NASBase.model.inception_arch import MNASSuperNet

from NASBase.duplication.dup_module import model_tracing


from NASBase.evo_search.utils import debug_get_net_info, sample_blk_choice_str
from NASBase.evo_search.mutation_operators import MutationOperator

from NASBase import multiprocessing_helper as mp_helper
from NASBase import file_utils, utils

from NASBase.evo_search.evo_memory import EvoMem, EvoMemTypes

from NASBase.hw_cost.Modules_inas_v1.CostModel import common
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



__all__ = ['EvolutionFinder']


#IX_EXPFACTOR = 0 
#IX_KSIZE = 1
#IX_NUMLAYERS = 2
#IX_SKIPSUPP = 3
#
# FOR INCEPTION
#IX_KERNEL_NUM_P1 = 0
#IX_KERNEL_NUM_P2 = 1
#IX_KERNEL_SIZE_P1 = 2
#IX_KERNEL_SIZE_P2 = 3
#IX_CH_RATIO = 4

if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    IX_STRIDE_FACTORS = 0
    IX_KERNEL_SIZES = 1
    IX_KERNEL_NUM = 2

elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    IX_STRIDE_FACTORS = 0
    IX_KERNEL_SIZES = 1
    IX_KERNEL_NUM = 2
    IX_MODULE_ABC = 3
    
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mobile':
    IX_EXPFACTOR = 0 
    IX_KERNEL_SIZES = 1
    IX_KERNEL_NUM = 2
    IX_SKIPSUPP = 3


#FOR SHUFFLE
#IX_STRIDE_FACTORS=0
#IX_KERNEL_SIZES=1
#IX_KERNEL_NUM=2


AVAILABLE_GPUIDS = [0, 1, 2, 3]
AVAILABLE_CPUIDS = np.arange(2)

USE_MULTIPROCESSING = True

if Settings.NAS_SETTINGS_GENERAL['MODE'] == 'none': 
    USE_TS = False
else:
    USE_TS = True

class ArchManager:
    def __init__(self, global_settings: Settings, dataset, net: MNASSuperNet):
     
        self.blk_choices = net.blk_choices
        self.num_blocks = net.num_blocks
        self.settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[global_settings.NAS_SETTINGS_GENERAL['DATASET']]
        
        # self.num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset].NUM_BLOCKS,       
    
    def all_samples(self):
        candidate_configs = [list(x) for x in itertools.product(self.blk_choices, repeat=self.num_blocks)]
        return candidate_configs
 
    def random_sample(self):
        choices_per_block = random.sample(self.blk_choices, self.num_blocks)
        return choices_per_block
     
        
    def random_resample(self, sample, bix):
        assert bix >= 0 and bix < self.num_blocks
        if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
            sample[bix][IX_STRIDE_FACTORS] = random.choice(self.settings_per_dataset['STRIDE_FACTORS'])
            sample[bix][IX_KERNEL_SIZES] = random.choice(self.settings_per_dataset['KERNEL_SIZES'])
            sample[bix][IX_KERNEL_NUM] = random.choice(self.settings_per_dataset['NUM_LAYERS_EXPLICIT'])
            
        elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
            sample[bix][IX_STRIDE_FACTORS] = random.choice(self.settings_per_dataset['STRIDE_FACTORS'])
            sample[bix][IX_KERNEL_SIZES] = random.choice(self.settings_per_dataset['KERNEL_SIZES'])
            sample[bix][IX_KERNEL_NUM] = random.choice(self.settings_per_dataset['NUM_LAYERS_EXPLICIT'])
            sample[bix][IX_MODULE_ABC] = random.choice(self.settings_per_dataset['MODULE_ABC'])

        elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mobile':
            sample[bix][IX_STRIDE_FACTORS] = random.choice(self.settings_per_dataset['STRIDE_FACTORS'])
            sample[bix][IX_KERNEL_SIZES] = random.choice(self.settings_per_dataset['KERNEL_SIZES'])
            sample[bix][IX_KERNEL_NUM] = random.choice(self.settings_per_dataset['NUM_LAYERS_EXPLICIT'])
            sample[bix][IX_SUPPORT_SKIP] = random.choice(self.settings_per_dataset['SUPPORT_SKIP'])

    
    # def random_resample(self, sample, i):
    #   assert i >= 0 and i < self.num_blocks
    #   sample['ks'][i] = random.choice(self.kernel_sizes)
    #   sample['e'][i] = random.choice(self.expand_ratios)

    # def random_resample_depth(self, sample, i):
    #   assert i >= 0 and i < self.num_stages
    #   sample['d'][i] = random.choice(self.depths)

    # def random_resample_resolution(self, sample):
    #   sample['r'][0] = random.choice(self.resolutions)





# all blocks are picked from the same set of choices
class ArchManagerDroppedSearchSpace(ArchManager):
    def __init__(self, global_settings: Settings, dataset, net: MNASSuperNet):
        super().__init__(global_settings, dataset, net)


        settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[global_settings.NAS_SETTINGS_GENERAL['DATASET']]
        # Not checking global_settings.TINAS['STAGE1_SETTINGS']['DROPPING_ENABLED'], as this class is used only for TINAS
        block_level_dropped_choices = global_settings.TINAS['STAGE1_SETTINGS']['DROPPING_BLOCK_LEVEL']
        
        if global_settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
            self.k_stride            = drop_choices(settings_per_dataset['STRIDE_FACTORS'], block_level_dropped_choices['STRIDE_FACTORS'])
            self.k_kernelsizes           = drop_choices(settings_per_dataset['KERNEL_SIZES'], block_level_dropped_choices['KERNEL_SIZES'])
            self.k_num_layers_explicit   = drop_choices(settings_per_dataset['NUM_LAYERS_EXPLICIT'], block_level_dropped_choices['NUM_LAYERS_EXPLICIT'])
       
        elif global_settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
            self.k_stride            = drop_choices(settings_per_dataset['STRIDE_FACTORS'], block_level_dropped_choices['STRIDE_FACTORS'])
            self.k_kernelsizes           = drop_choices(settings_per_dataset['KERNEL_SIZES'], block_level_dropped_choices['KERNEL_SIZES'])
            self.k_num_layers_explicit   = drop_choices(settings_per_dataset['NUM_LAYERS_EXPLICIT'], block_level_dropped_choices['NUM_LAYERS_EXPLICIT'])
            self.k_module   = drop_choices(settings_per_dataset['MODULE_ABC'], block_level_dropped_choices['MODULE_ABC'])
           
        elif global_settings.NAS_SETTINGS_GENERAL['ARC'] == 'mobile':
            self.k_stride            = drop_choices(settings_per_dataset['STRIDE_FACTORS'], block_level_dropped_choices['STRIDE_FACTORS'])
            self.k_kernelsizes           = drop_choices(settings_per_dataset['KERNEL_SIZES'], block_level_dropped_choices['KERNEL_SIZES'])
            self.k_num_layers_explicit   = drop_choices(settings_per_dataset['NUM_LAYERS_EXPLICIT'], block_level_dropped_choices['NUM_LAYERS_EXPLICIT'])
            self.k_support_skip   = drop_choices(settings_per_dataset['SUPPORT_SKIP'], block_level_dropped_choices['SUPPORT_SKIP'])
        

             #self.kernel_num_p2 = drop_choices(settings_per_dataset['KERNEL_NUM_P2'], block_level_dropped_choices['KERNEL_NUM_P2'])
        #self.kernel_size_p1 = drop_choices(settings_per_dataset['KERNEL_SIZE_P1'], block_level_dropped_choices['KERNEL_SIZE_P1'])
        #self.kernel_size_p2 = drop_choices(settings_per_dataset['KERNEL_SIZE_P2'], block_level_dropped_choices['KERNEL_SIZE_P2'])
        #self.ch_ratio = drop_choices(settings_per_dataset['IX_CH_RATIO'], block_level_dropped_choices['CH_RATIO'])


    def random_resample(self, sample, bix):
        assert bix >= 0 and bix < self.num_blocks

        if global_settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
            sample[bix][IX_STRIDE_FACTORS] = random.choice(self.k_stride)
            sample[bix][IX_KERNEL_SIZES] = random.choice(self.k_kernelsizes)
            sample[bix][IX_KERNEL_NUM] = random.choice(self.k_num_layers_explicit)
            
        elif global_settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
            sample[bix][IX_STRIDE_FACTORS] = random.choice(self.k_stride)
            sample[bix][IX_KERNEL_SIZES] = random.choice(self.k_kernelsizes)
            sample[bix][IX_KERNEL_NUM] = random.choice(self.k_num_layers_explicit)
            sample[bix][IX_MODULE_ABC] = random.choice(self.k_module)

        elif global_settings.NAS_SETTINGS_GENERAL['ARC'] == 'mobile':
            sample[bix][IX_STRIDE_FACTORS] = random.choice(self.k_stride)
            sample[bix][IX_KERNEL_SIZES] = random.choice(self.k_kernelsizes)
            sample[bix][IX_KERNEL_NUM] = random.choice(self.k_num_layers_explicit)
            sample[bix][IX_SUPPORT_SKIP] = random.choice(self.k_support_skip)


        
        #sample[bix][IX_KERNEL_SIZE_P1] = random.choice(self.kernel_size_p1)
        #sample[bix][IX_KERNEL_SIZE_P2] = random.choice(self.kernel_size_p2)
        #sample[bix][IX_CH_RATIO] = random.choice(self.ch_ratio)











class EvolutionFinder:
    valid_constraint_range = {
        'FLOPS': [150, 600],
        'MSP430': [10, 1000000000],
    }

    def __init__(self, global_settings, dataset, supernet, constraint_type, efficiency_constraint,
                 efficiency_predictor, accuracy_predictor, logfname, **kwargs):
     #imc_constraint,
        self.global_settings = global_settings
        self.dataset = dataset
        self.constraint_type = constraint_type
        self.exp_suffix = kwargs.get('exp_suffix', None) or self.global_settings.GLOBAL_SETTINGS['EXP_SUFFIX']
        self.logfname = logfname
        self.run_id = kwargs.get('run_id', None) or 0
  
        self.debug_enabled = self.global_settings.NAS_EVOSEARCH_SETTINGS['DEBUG_ENABLED']   # allows to disable some verbose prints
  
        if not constraint_type in self.valid_constraint_range.keys():
            #self.invite_reset_constraint_type()
            sys.exit("EvolutionFinder::Error: Invalid constraint type!")
   
        self.efficiency_constraint = efficiency_constraint
        if not (efficiency_constraint <= self.valid_constraint_range[constraint_type][1] and
                efficiency_constraint >= self.valid_constraint_range[constraint_type][0]):
            #self.invite_reset_constraint()
            sys.exit("EvolutionFinder::Error: Invalid constraint_value!")

        #self.imc_constraint = imc_constraint
        #if not (imc_constraint <= 100 and imc_constraint >= 0):
        #   sys.exit("EvolutionFinder::Error: Invalid imc_constraint value!")
        
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        block_search_space = global_settings.TINAS['STAGE2_SETTINGS']['BLOCK_SEARCH_SPACE']
        if block_search_space == 'dropped':
            self.arch_manager = ArchManagerDroppedSearchSpace(global_settings, global_settings.NAS_SETTINGS_GENERAL['DATASET'], supernet)
        elif block_search_space == 'default':
            self.arch_manager = ArchManager(global_settings, global_settings.NAS_SETTINGS_GENERAL['DATASET'], supernet)
        else:
            sys.exit("EvolutionFinder::Error: Invalid dropping strategy!")
        self.num_blocks = self.arch_manager.num_blocks
        self.net_choices = supernet.net_choices
        _, input_resolution = self.net_choices
        self.input_resolution = input_resolution
        #self.num_stages = self.arch_manager.num_stages

        # initialize caching
        self.evo_memory = EvoMem(self.global_settings.NAS_EVOSEARCH_SETTINGS, 
                                  net_choices = self.net_choices, 
                                  input_ch = self.global_settings.NAS_SETTINGS_PER_DATASET[self.dataset]['INPUT_CHANNELS'])
  
        # evo search settings
        # self.population_size = kwargs.get('population_size', 100)
        # self.max_time_budget = kwargs.get('max_time_budget', 500)
        # self.parent_ratio = kwargs.get('parent_ratio', 0.25)
        # self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)
        
        self.max_time_budget    = kwargs.get('max_time_budget', global_settings.NAS_EVOSEARCH_SETTINGS['GENERATIONS'])
        self.population_size    = kwargs.get('population_size', global_settings.NAS_EVOSEARCH_SETTINGS['POP_SIZE'])
        self.parent_ratio       = kwargs.get('parent_ratio', global_settings.NAS_EVOSEARCH_SETTINGS['PARENT_RATIO'])                
        self.mutation_ratio     = kwargs.get('mutation_ratio', global_settings.NAS_EVOSEARCH_SETTINGS['MUT_RATIO'])
        self.mutation_operator  = MutationOperator(global_settings, self.arch_manager, self.num_blocks)

        self.initial_population_fname = global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_INITIAL_POPULATION_FNAME']
        if not self.initial_population_fname:
            if not global_settings.NAS_SETTINGS_GENERAL['SEARCH_TIME_TESTING']:
                self.initial_population_fname = self.global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + self.exp_suffix + '_initial_population.json'


    # def invite_reset_constraint_type(self):
    #   print('Invalid constraint type! Please input one of:', list(self.valid_constraint_range.keys()))
    #   new_type = input()
    #   while new_type not in self.valid_constraint_range.keys():
    #       print('Invalid constraint type! Please input one of:', list(self.valid_constraint_range.keys()))
    #       new_type = input()
    #   self.constraint_type = new_type

    # def invite_reset_constraint(self):
    #   print('Invalid constraint_value! Please input an integer in interval: [%d, %d]!' % (
    #       self.valid_constraint_range[self.constraint_type][0],
    #       self.valid_constraint_range[self.constraint_type][1])
    #         )

    #   new_cons = input()
    #   while (not new_cons.isdigit()) or (int(new_cons) > self.valid_constraint_range[self.constraint_type][1]) or \
    #           (int(new_cons) < self.valid_constraint_range[self.constraint_type][0]):
    #       print('Invalid constraint_value! Please input an integer in interval: [%d, %d]!' % (
    #           self.valid_constraint_range[self.constraint_type][0],
    #           self.valid_constraint_range[self.constraint_type][1])
    #             )
    #       new_cons = input()
    #   new_cons = int(new_cons)
    #   self.efficiency_constraint = new_cons

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def check_constraints(self, sample):
        input_ch = self.global_settings.NAS_SETTINGS_PER_DATASET[self.dataset]['INPUT_CHANNELS']

        if not self.global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_BYPASS_EFFICIENCY']:
            
            # -- Results can be cached in a lookup table --
            cached_lat = self.evo_memory.query_tbl(sample, EvoMemTypes.LAT)
            #cached_imc = self.evo_memory.query_tbl(sample, EvoMemTypes.IMC)

            if None in [cached_lat]:    # if any of them are none, then recalculate  #, cached_imc
                efficiency = self.efficiency_predictor.predict_efficiency(sample, self.net_choices, input_ch)
            else:
                print(f'Cache hit for {str(sample)}')
                efficiency = cached_lat
                #imc = cached_imc    
       
            if efficiency == -1 or (self.efficiency_constraint != 0 and efficiency > self.efficiency_constraint):
                if self.debug_enabled: print(f'Skipping subnet - efficiency {efficiency} exceeds constraint {self.efficiency_constraint}')
                sample = None  # return None to indicate something wrong

            #if imc == -1 or (self.imc_constraint != 0 and imc > self.imc_constraint):
            #   if self.debug_enabled: print(f'Skipping subnet - IMC {imc} exceeds constraint {self.imc_constraint}')
            #   sample = None

            return sample, efficiency #, imc
        else:
            efficiency = -1 # = imc

            cached_nvm_fit = self.evo_memory.query_tbl(sample, EvoMemTypes.NVM_FIT)
            
            print("###check_constraints###")
            #print(self.net_choices)
            #print(sample)
            
            if USE_TS:
                tmp_dup_config = self.evo_memory.query_tbl(sample, EvoMemTypes.TS)
                under_mem= False
                
                if self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'pdq':
                    under_mem, peak_for_all, total_pdq_config, mac_count, access_gap, cached_nvm_fit = self.op_duplication(998,sample, 0, self.net_choices)

                    if not under_mem:
                        self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, 0)
                        return None, None #skip subnet, over constraint
                    else:
                        efficiency = 1
                        self.evo_memory.update_tbl(sample, EvoMemTypes.VM, peak_for_all)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, cached_nvm_fit)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.TS, total_pdq_config)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.MAC, mac_count)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.ACCESS, access_gap)
                        print("update_tbl: samples in evo")

                elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinyts':
                    under_mem, peak_after_sp, micrographs, mac_count, access_gap, cached_nvm_fit = self.op_duplication(998,sample, 0, self.net_choices)

                    if not under_mem:
                        self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, 0)
                        return None, None #skip subnet, over constraint
                    else:
                        efficiency = 1
                        self.evo_memory.update_tbl(sample, EvoMemTypes.VM, peak_after_sp)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, cached_nvm_fit)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.TS, micrographs)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.MAC, mac_count)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.ACCESS, access_gap)
                        print("update_tbl: samples in evo")
                
                elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinynas':
                    under_mem, peak_after_patch, min_lat_mem_result, mac_count, access_gap, cached_nvm_fit = self.op_duplication(998,sample, 0, self.net_choices)

                    if not under_mem:
                        self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, 0)
                        return None, None #skip subnet, over constraint
                    else:
                        efficiency = 1
                        self.evo_memory.update_tbl(sample, EvoMemTypes.VM, peak_after_patch)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, cached_nvm_fit)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.TS, min_lat_mem_result)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.MAC, mac_count)
                        self.evo_memory.update_tbl(sample, EvoMemTypes.ACCESS, access_gap)
                        print("update_tbl: samples in evo")


            else:
               cached_vm = self.evo_memory.query_tbl(sample, EvoMemTypes.VM)


            if cached_nvm_fit != None: # use cached value
                all_layers_fit_nvm = cached_nvm_fit
            else: # recalculate
                all_layers_fit_nvm = self.efficiency_predictor.predict_nvm_usage(sample, self.net_choices)
    
            if all_layers_fit_nvm:
                return sample, efficiency # imc
            else:
                if self.debug_enabled: print('Skipping subnet - not all layers fit NVM')
                return None, efficiency #, imc

    def random_sample(self):
        trials = 0
        while trials < 100:
            sample = self.arch_manager.random_sample()
            sample, efficiency = self.check_constraints(sample) #, imc
            if sample:
                return sample, efficiency #, imc
            trials += 1
            print(f'random_sample: retry {trials}')
        return None, None

    
    ##################################################################
    # EVO OPERATIONS 
    ##################################################################
 
 
    # mutate one or more of: resolution / block params (kernel_sz, exp_ratio) / depth per stage
    # same mutation probability used for all
    def mutate_sample(self, sample):
        while True:
            mutation_operator_type = self.global_settings.TINAS['STAGE2_SETTINGS']['MUTATION_OPERATOR']
            if mutation_operator_type == 'mutate_default':
                new_sample = self.mutation_operator.mutate_default(sample)
            elif mutation_operator_type == 'mutate_blockwise_prob':
                new_sample = self.mutation_operator.mutate_blockwise_prob(sample)
            elif mutation_operator_type == 'mutate_adaptive':
                new_sample = self.mutation_operator.mutate_adaptive(sample, self.best_valids)
            else:
                sys.exit("mutate_sample::Error: Invalid mutation operator!")
            
            # this mutation is only accepted if it passes the efficiency constraint
            new_sample, efficiency = self.check_constraints(new_sample)  #, imc
            if new_sample:
                return new_sample, efficiency#, imc


    def crossover_sample(self, sample1, sample2):     
        
        if (len(sample1) != len(sample2)):
            sys.exit("crossover_sample::Error - parents len mismatch!")
   
        while True:
            new_sample = copy.deepcopy(sample1)
   
            # for each block, we pick from either the parent 1 or parent 2
            for bix, blk in enumerate(sample1):
                new_sample[bix] = random.choice([sample1[bix], sample2[bix]])
       
            new_sample, efficiency = self.check_constraints(new_sample) #, imc
            if new_sample:
                return new_sample, efficiency #, imc

    # def crossover_sample(self, sample1, sample2):     
    #   constraint = self.efficiency_constraint
    #   while True:
    #       new_sample = copy.deepcopy(sample1)
    #       for key in new_sample.keys():
    #           if not isinstance(new_sample[key], list):
    #               continue
    #           for i in range(len(new_sample[key])):
    #               new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])

    #       efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
    #       if efficiency <= constraint:
    #           return new_sample, efficiency



    
 
    ##################################################################
    # MULTIPROCESSING WORKERS
    ##################################################################
    
    def _mpworker_pop_efficiency(self, worker_id, population_size, q, processed_subnets):
        print("_mpworker_pop_efficiency::Enter [{}]".format(worker_id))
        child_pool = []
        efficiency_pool = []
        #imc_pool = []
        while True:
            sample, efficiency= self.check_constraints(self.arch_manager.random_sample())
            if sample is not None:
                child_pool.append(sample); efficiency_pool.append(efficiency)
                q.put((sample, efficiency))
                with processed_subnets.get_lock():
                    processed_subnets.value += 1
            #print("_mpworker_pop_efficiency::Enter [{}][{}/{}]".format(worker_id, pix+1, batched_pop_size))
            with processed_subnets.get_lock():
                processed_subnets_value = processed_subnets.value
            print(f'Need {population_size} subnets, got {processed_subnets_value}')
            if processed_subnets_value >= population_size:
                break

        # for pix in range(batched_pop_size):
        #     sample, efficiency = self.random_sample()  #, imc
        #     if sample is None:
        #         break
        #     child_pool.append(sample); efficiency_pool.append(efficiency) #; imc_pool.append(imc)
        #     #print("_mpworker_pop_efficiency::Enter [{}][{}/{}]".format(worker_id, pix+1, batched_pop_size))
        # result = {
        #     "child_pool" : child_pool,
        #     "efficiency_pool" : efficiency_pool,
        #     #"imc_pool": imc_pool,
        # }  
        # return result

    def _mpworker_pop_accuracy(self, worker_id, batched_child_pool):
        print("_mpworker_pop_accuracy::Enter [{}], num_jobs={}".format(worker_id, len(batched_child_pool)))
  
        # check which samples are already cached in evo memory
        tmp_cached_accs={}
        for sample in batched_child_pool:           
            val_acc = self.evo_memory.query_tbl(sample, EvoMemTypes.ACC)
            if val_acc != None:
                k = sample_blk_choice_str(sample)
                tmp_cached_accs[k]=val_acc
        
        accs = self.accuracy_predictor.predict_accuracy(batched_child_pool, worker_id, input_resolution=self.input_resolution,
                                                        cached_accs=tmp_cached_accs) # cached accuracies will be used, not overwritten
        result = {  'batched_child_pool' : batched_child_pool,
                    'accs' : accs
                 }
        return result

    def _mpworker_pop_mutate(self, worker_id, population, parents_size, batched_mutation_numbers):
        print("_mpworker_pop_mutate::Enter [{}], num_jobs={}".format(worker_id, batched_mutation_numbers))
        child_pool = []
        efficiency_pool = []
        #imc_pool = []
        for i in range(batched_mutation_numbers):
            par_sample = population[np.random.randint(parents_size)][1]
            
            # Mutate
            new_sample, efficiency = self.mutate_sample(par_sample) #, imc
            child_pool.append(new_sample); efficiency_pool.append(efficiency) #; imc_pool.append(imc)
            #print("_mpworker_pop_mutate::Enter [{}][{}/{}]".format(worker_id, i+1, batched_mutation_numbers))
        result = {
                "child_pool" : child_pool, "efficiency_pool" : efficiency_pool, #"imc_pool": imc_pool,
        }  
        return result

    def _mpworker_pop_crossover(self, worker_id, population, parents_size, batched_crossover_numbers):
        print("_mpworker_pop_crossover::Enter [{}], num_jobs={}".format(worker_id, batched_crossover_numbers))
        child_pool = []
        efficiency_pool = []
        #imc_pool = []
        for i in range(batched_crossover_numbers):
            par_sample1 = population[np.random.randint(parents_size)][1]    # could this give two identical parents ?
            par_sample2 = population[np.random.randint(parents_size)][1]

            # Crossover
            new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)  #, imc
            child_pool.append(new_sample); efficiency_pool.append(efficiency) #; imc_pool.append(imc)
            #print("_mpworker_pop_crossover::Enter [{}][{}/{}]".format(worker_id, i+1, batched_crossover_numbers))
        result = {
                "child_pool" : child_pool, "efficiency_pool" : efficiency_pool,# "imc_pool": imc_pool,
        }  
        return result
    
    ##################################################################
    # LOGGING RELATED
    ##################################################################  
    def log_progress(self, iteration, population):
        best_subnet = population[0]
        worst_subnet = population[-1]
        best_acc, _, best_efficiency= best_subnet  #, best_imc 
        worst_acc, _, worst_efficiency = worst_subnet  #, worst_imc
        best_info_dict = debug_get_net_info(best_subnet)
        worst_info_dict = debug_get_net_info(worst_subnet)
        best_score = self.get_score(best_subnet)
        worst_score = self.get_score(worst_subnet)

        #pprint(population)
  
        field_names = ["iter", "best_score", "worst_score", "best_acc", "worst_acc", "best_efficiency", "worst_efficiency", "best_config", "worst_config", "uniq"] #"best_imc", "worst_imc",
        field_values = [iteration, best_score, worst_score, best_acc, worst_acc, best_efficiency, worst_efficiency, best_info_dict["config"], worst_info_dict["config"], self.calc_unique_subnets(population)]
        assert len(field_names) == len(field_values)  #best_imc, worst_imc,

        logger = utils.CsvLogger((self.global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] +
                                  self.exp_suffix + "_evo_search.csv"),
                                 field_names)
        logger.log(field_values)

        rlog = None
        if self.global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER']:
            rlog = get_remote_logger_obj(self.global_settings)
        if rlog:
            rlog.log({
                field_name: field_value
                for field_name, field_value in zip(field_names, field_values)
            })
 
    def get_score(self, item):
        acc, child, efficiency = item  #, imc
        score_type = self.global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_SCORE_TYPE']

        Lreq = self.global_settings.PLATFORM_SETTINGS['LAT_E2E_REQ']

        if score_type == 'ACC_IMC':
            return acc * (1/imc)
        elif score_type == 'ACC':
            return acc
        elif score_type == 'ACC_IMO_LREQ':
            return acc * (1/imc) * (efficiency/Lreq)
        elif score_type == 'ACC_LREQ':
            return acc * (efficiency/Lreq)
        else:
            sys.exit("get_score: Invalid score type!")

    @staticmethod
    def calc_unique_subnets(population):
        # use strings as subnet configs cannot be used in a set
        subnet_strings = [str(s) for s in population]
        return len(set(subnet_strings))

    def init_population(self, verbose=False):
        population_size = self.population_size

        population = []  # (validation, sample, latency, imc) tuples
        child_pool = []
        efficiency_pool = []
        #imc_pool = []
        if verbose:
            print('Generate random population...')
    
        # ==== get latency ====
        print("init population : latency")
        if USE_MULTIPROCESSING: # (### MULTIPROCESSING: CPU)
            num_workers = self.global_settings.NAS_EVOSEARCH_SETTINGS['FIXED_NUM_CPU_WORKERS']
            
            mp = multiprocessing.get_context('spawn')
            processed_subnets = mp.Value('i', 0)
            q = mp.SimpleQueue()
            ctx = torch.multiprocessing.spawn(self._mpworker_pop_efficiency,
                args=(population_size, q, processed_subnets),
                nprocs=num_workers,
                join=False)
            while len(child_pool) < population_size:
                sample, efficiency = q.get()
                child_pool.append(sample)
                efficiency_pool.append(efficiency)
            # Loop on join until it returns True or raises an exception.
            while not ctx.join():
                pass

            # if (population_size % num_workers) > 0:
            #     sys.exit("run_evolution_search::Error - init get latency - non divisible num workers: {},{}".format(num_workers, population_size))
            # batched_pop_size = int(np.ceil(population_size/num_workers))            
            # all_worker_results = mp_helper.run_multiprocessing_workers(
            #     num_workers=num_workers,
            #     worker_func= self._mpworker_pop_efficiency,
            #     worker_type='CPU',
            #     common_args=(batched_pop_size,), worker_args=(),
            # )           
            # # combine results
            # for worker_result in all_worker_results:
            #     child_pool.extend(worker_result['child_pool'])
            #     efficiency_pool.extend(worker_result['efficiency_pool'])
                #imc_pool.extend(worker_result['imc_pool'])
        else:  
            for pix in range(population_size):
                sample, efficiency = self.random_sample()  #, imc
                child_pool.append(sample)
                efficiency_pool.append(efficiency)
                #imc_pool.append(imc)
    
        child_pool = list(filter(None, child_pool))
        if len(child_pool) < population_size:
            sys.exit("init population : unable to find enough subnets")

        # ==== get accuracy ====
        print("init population : accuracy")
        if USE_MULTIPROCESSING: # (### MULTIPROCESSING: GPU)
            num_workers = mp_helper.get_max_num_workers('GPU')
            if (len(child_pool) % num_workers) > 0:
                sys.exit("run_evolution_search::Error - init get accuracy - non divisible num workers: {},{}".format(num_workers, len(child_pool)))
            else:
                #batched_child_pool = np.array_split(child_pool, num_workers)           
                batched_child_pool = list(split_list_chunks(child_pool, num_workers))
                print('batched_child_pool', batched_child_pool)
            all_worker_results = mp_helper.run_multiprocessing_workers(
                num_workers=num_workers,
                worker_func= self._mpworker_pop_accuracy,
                worker_type='GPU',
                common_args=(), worker_args=(batched_child_pool),
            )   
            # combine results
            accs = list(itertools.chain.from_iterable(worker_result['accs'] for worker_result in all_worker_results))
        else:
            accs = self.accuracy_predictor.predict_accuracy(child_pool, mp_helper.available_gpus()[0], input_resolution=self.input_resolution)

        print('Number of unique subnets in the initial population:', self.calc_unique_subnets(population))
   
        # -- create population
        for pix in range(population_size):
            population.append([accs[pix], child_pool[pix], efficiency_pool[pix]])  #,imc_pool[pix]
  
        
        #self._debug_dump_pop_info(population)

        # -- update evo mem
        for sample, lat, acc in zip(child_pool, efficiency_pool, accs):  #, imc, , imc_pool
            self.evo_memory.update_tbl_multival(sample, 
                                                [EvoMemTypes.LAT,  EvoMemTypes.ACC], 
                                                [lat, acc]
                                                )
            #EvoMemTypes.IMC,imc,

        return population


    ##################################################################
    # duplication
    ##################################################################  
    def op_duplication(self, idx, child, lat, supernet_config):
        
        width_multiplier, input_resolution = supernet_config

        # for blk_choices of each child: map value into list 
        blk_choices_list = parametric_supernet_blk_choices(global_settings=self.global_settings)
        subnet_choice_per_blk = blkchoices_to_blkchoices_ixs(blk_choices_list, child)
        subnet = get_subnet(self.global_settings, self.dataset, blk_choices_list, subnet_choice_per_blk, -1, width_multiplier, input_resolution)

        #print("The 1st child, shown in subnet format:")
        #print(subnet)

        net_input = get_dummy_net_input_tensor(self.global_settings, input_resolution)
        subnet_dims = get_network_dimension(subnet, input_tensor = net_input)
        subnet_obj = get_network_obj(subnet_dims)
        performance_model = PlatPerf(self.global_settings.NAS_SETTINGS_GENERAL, self.global_settings.PLATFORM_SETTINGS)
        _, network_nvm_usage, _ = performance_model.get_nvm_usage(subnet_obj)
        max_features = max(f for f, _ in network_nvm_usage)
        cached_nvm_fit = 0
        if max_features <= self.global_settings.PLATFORM_SETTINGS["NVM_CAPACITY"]:
            cached_nvm_fit = 1
        else:
            cached_nvm_fit = 0


        vm_available = self.global_settings.NAS_SETTINGS_GENERAL['VMSIZE']
        print("Convert pth to onnx:")
        onnx_path = self.global_settings.NAS_EVOSEARCH_SETTINGS['ONNX_FILE_PATH']
        model_name = 'sso_sample_'+str(idx)
        input_names = ["input"]
        output_names = ["output"]
        under_mem = False
        onnx_file = os.path.join(onnx_path, model_name + '.onnx')

        try:
            # Export the ONNX model
            torch.onnx.export(
                subnet,
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
            
            return False, None, None, None, None, cached_nvm_fit   # skip this model
        if not os.path.exists(onnx_file) or os.path.getsize(onnx_file) < 10000:
            print(f"[*] ONNX file not created properly or is too small: {onnx_file}")
            return False, None, None, None, None, cached_nvm_fit   # skip this model

        try:
            onnx_model = onnx.load(onnx_file)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print(f"[*] ONNX file load/check failed: {onnx_file}")
            print(f"[*] Error: {e}")
            return False, None, None, None, None, cached_nvm_fit   # skip this model        
        # check created model
        #onnx_model = onnx.load('new_onnx/resnet-erasing.onnx')
        #onnx.checker.check_model(onnx_model)
        under_mem = False
        
        #---- call model_tracing by onnx ----#
        if self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'pdq':
            which_over_mem, total_latency, under_mem, total_pdq_config, duration, time_record, remaining_peaks, mac_count, access_gap, max_mem_without_dup = model_tracing(onnx_path, model_name)
            
            if not under_mem:
                print("can not find valid duplication")
                return False, None, None, None, None, cached_nvm_fit   # skip this model

            peak_after_dup = 0
            if total_pdq_config:        
                for idx, entry in enumerate(total_pdq_config):
                    if entry['after_peak_mem'] > peak_after_dup:
                        peak_after_dup = entry['after_peak_mem']
                    
            peak_for_all = max(max_mem_without_dup, peak_after_dup)

            if peak_for_all > (vm_available*1024):
                print("after TS, still over vm")
                under_mem = False
                return False, None, None, None, None, cached_nvm_fit
            else:
                under_mem = True

            return under_mem, peak_for_all, total_pdq_config, mac_count, access_gap, cached_nvm_fit
                

        elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinyts':
            which_over_mem, duplicated_per_micro, split_ts_per_micro, micrographs, peak_mem_per_micro, inner_time, unsplittable_op_indices, maxone_unsp, mac_count, access_gap = model_tracing(onnx_path, model_name)
                    
            peak_after_sp=0
            for m_id, micro in enumerate(micrographs):
                micro_peak, micro_mode, micro_sp = peak_mem_per_micro[m_id]

                if micro_peak > peak_after_sp:
                    peak_after_sp = micro_peak
                    
            if maxone_unsp > peak_after_sp:
                peak_after_sp = maxone_unsp


            if peak_after_sp > (vm_available*1024):
                print("after tinyts, can not meet vm constraint")
                return False, None, None, None, None, cached_nvm_fit
            else:
                under_mem = True

            return under_mem, peak_after_sp, micrographs, mac_count, access_gap, cached_nvm_fit

        elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinynas':  ## wait for new version
            which_over_mem, min_mem_result, min_latency_result, tried_but_not_valid = model_tracing(onnx_path, model_name) 
                    

            if min_latency_result:
                peak_after_patch = min_latency_result['total_peak_mem']
                mac_count = min_latency_result['mac_ori'] + min_latency_result['mac_gap']+min_latency_result['mac_other']
                access_gap = min_latency_result['access_gap'] / (min_latency_result['access_ori'] + min_latency_result['access_other'])
                
                if peak_after_patch > (vm_available*1024):
                    print("after tinyts, can not meet vm constraint")
                    return False, None, None, None, None, cached_nvm_fit
                else:
                    under_mem = True

                return under_mem, peak_after_patch, min_latency_result, mac_count, access_gap, cached_nvm_fit

            elif min_mem_result:
                peak_after_patch = min_mem_result['total_peak_mem']
                mac_count = min_mem_result['mac_ori']+min_mem_result['mac_gap']+min_mem_result['mac_other']
                access_gap = min_mem_result['access_gap'] / (min_mem_result['access_ori'] + min_mem_result['access_other'])

                if peak_after_patch > (vm_available*1024):
                    print("after tinyts, can not meet vm constraint")
                    return False, None, None, None, None, cached_nvm_fit
                else:
                    under_mem = True

                return under_mem, peak_after_patch, min_mem_result, mac_count, access_gap, cached_nvm_fit

            else:
                print("didn't find valid patching")
                return False, None, None, None, None, cached_nvm_fit

            

        #total_lan, under_mem_TF ,dup_path, est_peak_per_path, q_list_per_path = model_tracing(onnx_path, onnx_name)
        
        
        #print("The 1st child, shown in subnet_obj format:")
        #print(subnet_obj)
        #print("The 1st child, dims of layers: type, IFM, OFM, K, stride, padding")

        #print(subnet_obj[0]['optype'])
        #print(subnet_obj[0]['IFM'].n, subnet_obj[0]['IFM'].ch, subnet_obj[0]['IFM'].h, subnet_obj[0]['IFM'].w)
        #print(subnet_obj[0]['OFM'].n, subnet_obj[0]['OFM'].ch, subnet_obj[0]['OFM'].h, subnet_obj[0]['OFM'].w)
        #print(subnet_obj[0]['K'].n, subnet_obj[0]['K'].ch, subnet_obj[0]['K'].h, subnet_obj[0]['K'].w)
        #print(subnet_obj[0]['stride'], subnet_obj[0]['pad'])



    
    ##################################################################
    # MAIN EVO LOOP
    ##################################################################  
 
    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))

        # ========================= INITIALIZE RANDOM POPULATION
        self.best_valids = best_valids = [-100]
        best_info = None
        population = []  # (validation, sample, latency, imc) tuples

        # TODO: load initial population from dump_pop
        if self.initial_population_fname and os.path.exists(self.initial_population_fname):
            initial_population = file_utils.json_load(self.initial_population_fname)
            population = copy.deepcopy(initial_population)
            print ("===> Initial population loaded from - ", self.initial_population_fname)
        else:
            print ("===> Initial population file %s not found, generating a new one" % (self.initial_population_fname,))

        if not population:
            population = self.init_population(verbose)

        if self.initial_population_fname:
            file_utils.json_dump(self.initial_population_fname, population)
            print ("===> Dumping initial population to - ", self.initial_population_fname)
  
 
        # ========================= START EVOLUTION
        if verbose:
            print('Start Evolution...')
            
   
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(range(max_time_budget), desc='Searching with %s constraint (%s)' % (self.constraint_type, self.efficiency_constraint)):
            
            population = sorted(population, key=self.get_score, reverse=True)   # sort by accuracy desc order

            self.log_progress(iter, population)
            self._debug_dump_pop_info(population, dump_onscreen=False,
                                        dump_fname=(self.global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + self.exp_suffix + "_dump_pop.json")
                                    ) 

            parents = population[:parents_size]  # get parent samples
   
            
  
            best_parent = parents[0]
            best_acc = best_parent[0]   # accuracy of best parent
            best_score = self.get_score(best_parent)
            if verbose:
                print('[{}] Iter: {} Acc: {} Score: {}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 
                                                    iter, best_acc, best_score))
                

            # update best accuracy in each iteration, if same, then just add the same
            if best_score > best_valids[-1]:
                best_valids.append(best_score)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            
            # initialize new parents and empty children list for this iteration
            population = parents    # parents are top N candidates in this population
            child_pool = []
            efficiency_pool = []
            #imc_pool = []
   
            # ========================= MUTATION
            # mutate a fixed number of random candidates (### MULTIPROCESSING: CPU)
            # populate children list using mutated candidates from current population
            if USE_MULTIPROCESSING: # (### MULTIPROCESSING: CPU)
                num_workers = self.global_settings.NAS_EVOSEARCH_SETTINGS['FIXED_NUM_CPU_WORKERS']
                if (mutation_numbers % num_workers) > 0:
                    sys.exit("run_evolution_search::Error - mutation - non divisible num workers: {},{}".format(num_workers, mutation_numbers))
                else:
                    batched_mutation_numbers = int(np.ceil(mutation_numbers/num_workers))
        
                all_worker_results = mp_helper.run_multiprocessing_workers(
                    num_workers=num_workers,
                    worker_func= self._mpworker_pop_mutate,
                    worker_type='CPU',
                    common_args=(population, parents_size,batched_mutation_numbers,), worker_args=(),
                )           
                # combine results
                for worker_result in all_worker_results:
                    child_pool.extend(worker_result['child_pool'])
                    efficiency_pool.extend(worker_result['efficiency_pool'])
                    #imc_pool.extend(worker_result['imc_pool'])
            else:
                for i in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]                 
                    # Mutate
                    new_sample, efficiency = self.mutate_sample(par_sample) #, imc
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)
                    #imc_pool.append(imc)

            # -- update evo mem
            for sample, lat in zip(child_pool, efficiency_pool): #, imc
                self.evo_memory.update_tbl_multival(sample, 
                                                    [EvoMemTypes.LAT],
                                                    [lat]
                                                    )
            #, EvoMemTypes.IMC , imc, , imc_pool
            
            # ========================= CROSSOVER
            # ---------- crossover a fixed number of random candidates  (### MULTIPROCESSING: CPU)
            crossover_numbers = population_size - mutation_numbers
            if USE_MULTIPROCESSING: # (### MULTIPROCESSING: CPU)
                num_workers = self.global_settings.NAS_EVOSEARCH_SETTINGS['FIXED_NUM_CPU_WORKERS']
                if (crossover_numbers % num_workers) > 0:
                    sys.exit("run_evolution_search::Error - crossover - non divisible num workers: {},{}".format(num_workers, crossover_numbers))
                else:
                    batched_crossover_numbers = int(np.ceil(crossover_numbers/num_workers))
                all_worker_results = mp_helper.run_multiprocessing_workers(
                    num_workers=num_workers,
                    worker_func= self._mpworker_pop_crossover,
                    worker_type='CPU',
                    common_args=(population, parents_size,batched_crossover_numbers,), worker_args=(),
                )           
                # combine results
                for worker_result in all_worker_results:
                    child_pool.extend(worker_result['child_pool'])
                    efficiency_pool.extend(worker_result['efficiency_pool'])
                    #imc_pool.extend(worker_result['imc_pool'])
            else:
                for i in range(crossover_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]    # possible identical parents ?
                    par_sample2 = population[np.random.randint(parents_size)][1]
        
                    # Crossover
                    new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2) #, imc
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)
                    #imc_pool.append(imc)


            # -- update evo mem   #, imc, imc_pool, EvoMemTypes.IMC, imc
            for sample, lat in zip(child_pool, efficiency_pool):
                self.evo_memory.update_tbl_multival(sample, 
                                                    [EvoMemTypes.LAT],
                                                    [lat]
                                                    )
            # ========================= GET ACCURACY FOR NEW POP
            if USE_MULTIPROCESSING: # (### MULTIPROCESSING: CPU)
                num_workers = mp_helper.get_max_num_workers('GPU')                              
                if (len(child_pool) % num_workers) > 0:
                    sys.exit("run_evolution_search::Error - final accuracy - non divisible num workers: {},{}".format(num_workers, len(child_pool)))
                else:
                    batched_child_pool = np.array_split(child_pool, num_workers)
                all_worker_results = mp_helper.run_multiprocessing_workers(
                    num_workers=num_workers,
                    worker_func= self._mpworker_pop_accuracy,
                    worker_type='GPU',
                    common_args=(), worker_args=(batched_child_pool),                   
                )
                # combine results
                accs = list(itertools.chain.from_iterable(worker_result['accs'] for worker_result in all_worker_results))
            else:
                accs = self.accuracy_predictor.predict_accuracy(child_pool, mp_helper.available_gpus()[0], input_resolution=self.input_resolution)


            # -- duplication
            
            under_mem=False
            cached_nvm_fit = 0

            if USE_TS:
                print("Start Duplication.....")
                input_ch = self.global_settings.NAS_SETTINGS_PER_DATASET[self.dataset]['INPUT_CHANNELS']

                #child = child_pool[0]
                print("Test the child dup: call dup func.")
                for idx, child in enumerate(child_pool):
                    print(idx, child)

                    lat = efficiency_pool[idx]
                    under_mem = False
                    
                    if self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'pdq':
                        under_mem, peak_for_all, total_pdq_config, mac_count, access_gap, cached_nvm_fit = self.op_duplication(idx,sample, lat, self.net_choices)

                        if not under_mem:
                            print("TS doesn't help! ") #skip subnet, over constraint
                            self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, 0)
                        else:
                            #efficiency = 1
                            self.evo_memory.update_tbl(sample, EvoMemTypes.VM, peak_for_all)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, cached_nvm_fit)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.TS, total_pdq_config)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.MAC, mac_count)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.ACCESS, access_gap)
                            print("update_tbl: samples in evo")

                    elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinyts':
                        under_mem, peak_after_sp, micrographs, mac_count, access_gap, cached_nvm_fit = self.op_duplication(idx,sample, lat, self.net_choices)

                        if not under_mem:
                            print("TS doesn't help! ") #skip subnet, over constraint
                            self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, 0)
                        else:
                            #efficiency = 1
                            self.evo_memory.update_tbl(sample, EvoMemTypes.VM, peak_after_sp)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, cached_nvm_fit)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.TS, micrographs)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.MAC, mac_count)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.ACCESS, access_gap)
                            print("update_tbl: samples in evo")
                    
                    elif self.global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinynas':
                        under_mem, peak_after_patch, min_lat_mem_result, mac_count, access_gap, cached_nvm_fit = self.op_duplication(idx,sample, lat, self.net_choices)

                        if not under_mem:
                            print("TS doesn't help! ") #skip subnet, over constraint
                            self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, 0)
                        else:
                            #efficiency = 1
                            self.evo_memory.update_tbl(sample, EvoMemTypes.VM, peak_after_patch)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.NVM_FIT, cached_nvm_fit)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.TS, min_lat_mem_result)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.MAC, mac_count)
                            self.evo_memory.update_tbl(sample, EvoMemTypes.ACCESS, access_gap)
                            print("update_tbl: samples in evo")
                    
                    # if under_mem:
                    #     print("Test the child dup: finished!")
                    #     efficiency_pool[idx]=total_lan
                            
                    # else:
                    #     print("can not duplication! ")
                        


            else:
                print("NO USE TS ")

            # -- create population
            for i in range(population_size):
                population.append([accs[i], child_pool[i], efficiency_pool[i]])  #, imc_pool[i]
                # log best and worst accs at the end of the generation
    
                # -- update evo mem
                self.evo_memory.update_tbl(child_pool[i], EvoMemTypes.ACC, accs[i])
    
            cur_logfname = self.logfname.replace('.json', f'-{self.run_id}-gen{iter}.json')

            best_info_dict = self.get_metadata_for_best_solution(best_info)

            

            # TODO: measure time taken for each generation
            time_taken = None
            best_solution = best_valids, best_info_dict, time_taken

            file_utils.delete_file(cur_logfname)
            file_utils.json_dump(cur_logfname, best_solution)
            

        print('Number of unique subnets after mutation and crossover:', self.calc_unique_subnets(population))


            # XXX: check the number of unique candidates in the population

        return best_valids, best_info_dict

    def get_metadata_for_best_solution(self, best_info):
        acc, child, efficiency = best_info #, imc

        subnet_latency_info = self.efficiency_predictor.predict_network_latency_verbose(child, self.net_choices)
        print("subnet_latency_info:", subnet_latency_info)  # Debugging output

        perf_exec_design = subnet_latency_info["perf_exec_design_contpow_fp"]
        print("perf_exec_design:", perf_exec_design)
        if perf_exec_design is None:
            print("Warning: perf_exec_design_contpow_fp is None!")
            perf_exec_design = []  # Prevents crash

        # Extract individual design parameters (tile sizes, ...)
        for layer in perf_exec_design:
            Tr, Tc, Tm, Tn, reuse_sch, S = common.string_to_params_all(layer['params'])
            layer['params'] = {'Tr': Tr, 'Tc': Tc, 'Tm': Tm, 'Tn': Tn, 'reuse_sch': reuse_sch, 'S': S}

        subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(self.arch_manager.blk_choices, child)

        performance_model = PlatPerf(self.global_settings.NAS_SETTINGS_GENERAL, self.global_settings.PLATFORM_SETTINGS)
        subnet_obj, _ = get_subnet_from_config(self.global_settings, self.dataset, child, self.net_choices)
        _, network_nvm_usage, _ = performance_model.get_nvm_usage(subnet_obj)

        return {
            # "subnet_name": xxx,
            "subnet_choice_per_blk": child,
            "subnet_choice_per_blk_ixs": subnet_choice_per_blk_ixs,
            "supernet config": self.net_choices,
            "subnet_latency_info": subnet_latency_info,
            "network_nvm_usage": network_nvm_usage,
            "lat_contpow": efficiency,
            # "lat_contpow": xxx,
            "accuracy": acc,
            #"imc": imc,
            "score": self.get_score(best_info),
        }
    
    
 
    
    ##################################################################
    # DEBUG RELATED
    ##################################################################
    def _debug_dump_pop_info(self, population, dump_onscreen=True, dump_fname=None):        
        if (dump_onscreen):
            print("====== POPULATION DUMP: START ==================================")       
            for sample in population:      
                pprint(sample)
            print("====== POPULATION DUMP: END   ==================================")
   
        # dump to a file
        if dump_fname != None:
            if (file_utils.file_exists(dump_fname)):
                dump_json_data = file_utils.json_load(dump_fname)
                dump_json_data.append(population)
                file_utils.json_dump(dump_fname, dump_json_data)
            
            else:
                dump_json_data = [population]
                file_utils.json_dump(dump_fname, dump_json_data)
        else:
            pass
       
   

        
      
      
      
    
def op_duplication_subnet(global_settings, net_input, subnet):
    
    
  
    subnet_dims = get_network_dimension(subnet, input_tensor = net_input)
    subnet_obj = get_network_obj(subnet_dims)

    vm_available = global_settings.NAS_SETTINGS_GENERAL['VMSIZE']
    print("Convert pth to onnx:")
    onnx_path = global_settings.NAS_EVOSEARCH_SETTINGS['ONNX_FILE_PATH']
    model_name = 'sample_subnet'
    input_names = ["input"]
    output_names = ["output"]
    under_mem = False
    onnx_file = os.path.join(onnx_path, model_name + '.onnx')

    try:
        # Export the ONNX model
        torch.onnx.export(
            subnet,
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
        
        return False, None, None, None, None   # skip this model
    if not os.path.exists(onnx_file) or os.path.getsize(onnx_file) < 10000:
        print(f"[*] ONNX file not created properly or is too small: {onnx_file}")
        return False, None, None, None, None   # skip this model
        
    try:
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"[*] ONNX file load/check failed: {onnx_file}")
        print(f"[*] Error: {e}")
        return False, None, None, None, None   # skip this model        
        # check created model
        #onnx_model = onnx.load('new_onnx/resnet-erasing.onnx')
        #onnx.checker.check_model(onnx_model)
    under_mem = False
        
        #---- call model_tracing by onnx ----#
    if global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'pdq':
        which_over_mem, total_latency, under_mem, total_pdq_config, duration, time_record, remaining_peaks, mac_count, access_gap, max_mem_without_dup = model_tracing(onnx_path, model_name)
            
        if not under_mem:
            print("can not find valid duplication")
            return False, None, None, None, None   # skip this model
            peak_after_dup = 0
        if total_pdq_config:        
            for idx, entry in enumerate(total_pdq_config):
                if entry['after_peak_mem'] > peak_after_dup:
                    peak_after_dup = entry['after_peak_mem']
                    
        peak_for_all = max(max_mem_without_dup, peak_after_dup)

        if peak_for_all > (vm_available*1024):
            print("after TS, still over vm")
            under_mem = False
            return False, None, None, None, None
        else:
            under_mem = True

        return under_mem, peak_for_all, total_pdq_config, mac_count, access_gap
                

    elif global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinyts':
        which_over_mem, duplicated_per_micro, split_ts_per_micro, micrographs, peak_mem_per_micro, inner_time, unsplittable_op_indices, maxone_unsp, mac_count, access_gap = model_tracing(onnx_path, model_name)
                
        peak_after_sp=0
        for m_id, micro in enumerate(micrographs):
            micro_peak, micro_mode, micro_sp = peak_mem_per_micro[m_id]

            if micro_peak > peak_after_sp:
                peak_after_sp = micro_peak
                    
        if maxone_unsp > peak_after_sp:
            peak_after_sp = maxone_unsp


        if peak_after_sp > (vm_available*1024):
            print("after tinyts, can not meet vm constraint")
            return False, None, None, None, None
        else:
            under_mem = True

        return under_mem, peak_after_sp, micrographs, mac_count, access_gap

    elif global_settings.NAS_SETTINGS_GENERAL['MODE'] == 'tinynas':  ## wait for new version
        which_over_mem, min_mem_result, min_latency_result, tried_but_not_valid = model_tracing(onnx_path, model_name) 
                    

        if min_latency_result:
            peak_after_patch = min_latency_result['total_peak_mem']
            mac_count = min_latency_result['mac_ori'] + min_latency_result['mac_gap']+min_latency_result['mac_other']
            access_gap = min_latency_result['access_gap'] / (min_latency_result['access_ori'] + min_latency_result['access_other'])
                
            if peak_after_patch > (vm_available*1024):
                print("after tinyts, can not meet vm constraint")
                return False, None, None, None, None
            else:
                under_mem = True

            return under_mem, peak_after_patch, min_latency_result, mac_count, access_gap

        elif min_mem_result:
            peak_after_patch = min_mem_result['total_peak_mem']
            mac_count = min_mem_result['mac_ori']+min_mem_result['mac_gap']+min_mem_result['mac_other']
            access_gap = min_mem_result['access_gap'] / (min_mem_result['access_ori'] + min_mem_result['access_other'])

            if peak_after_patch > (vm_available*1024):
                print("after tinyts, can not meet vm constraint")
                return False, None, None, None, None
            else:
                under_mem = True

            return under_mem, peak_after_patch, min_mem_result, mac_count, access_gap

        else:
            print("didn't find valid patching")
            return False, None, None, None, None


