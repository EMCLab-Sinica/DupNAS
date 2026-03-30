""" Contains hyper-parameters for the evolutionary search process    
"""
import os, sys
import torch
import numpy as np
import time
import math
import copy
from os.path import dirname, realpath
from pprint import pprint

from .evolution_finder import EvolutionFinder
from .accuracy_predictor import AccuracyPredictor
from .latency_estimator import LatencyEstimator


#sys.path.append("../..")
#sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from NASBase import utils as utils
from NASBase import file_utils as file_utils
from NASBase.model.common_utils import get_supernet
from settings import Settings, arg_parser, load_settings







# cuda_available = torch.cuda.is_available()
# if cuda_available:
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     torch.cuda.manual_seed(random_seed)
#     print('Using GPU.')
# else:
#     print('Using CPU.')
#print('Using CPU')





def evo_search(global_settings: Settings, dataset, supernet, logfname, run_id=0, exp_suffix=None):
    
    # Check if supernet is the best one
    print('Supernet net_choices: {}'.format(supernet.net_choices))
    

    latency_estimator = LatencyEstimator(global_settings=global_settings)   

    # accuracy predictor
    accuracy_predictor = AccuracyPredictor(
        global_settings,
        supernet, 
    )

    latency_constraint = global_settings.PLATFORM_SETTINGS['LAT_E2E_REQ']  
     
    P = global_settings.NAS_EVOSEARCH_SETTINGS['POP_SIZE']  # The size of population in each generation
    N = global_settings.NAS_EVOSEARCH_SETTINGS['GENERATIONS']  # How many generations of population to be searched
    r = global_settings.NAS_EVOSEARCH_SETTINGS['PARENT_RATIO']  # The ratio of networks that are used as parents for next generation
    params = {
        'global_settings' : global_settings,
        'exp_suffix': exp_suffix,
        'dataset' : dataset,
        'supernet': supernet,        
        'efficiency_constraint': latency_constraint,
        'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': latency_estimator, # To use a predefined efficiency predictor.
        'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,        
        'logfname': logfname,
        'run_id': run_id,
    }

    # build the evolution finder
    finder = EvolutionFinder(**params)

    # start searching
    result_lis = []
    st = time.time()
    best_valids, best_info = finder.run_evolution_search(verbose=True)
    result_lis.append(best_info)
    ed = time.time()
    time_taken = ed-st
    print('Found best architecture with latency <= %.2f ms in %.2f seconds! '
        'It achieves %.2f%s predicted accuracy with %.2f ms latency.' %
        (latency_constraint, time_taken, best_info['accuracy'], '%', best_info['lat_contpow']))

    # visualize the architecture of the searched sub-net
    print('Architecture of the searched sub-net:')
    pprint(best_info['subnet_choice_per_blk'])
 
    return best_valids, best_info, time_taken



if __name__ == '__main__':
    
    # get settings  (global)
    global_settings = Settings() # default settings
    global_settings = arg_parser(global_settings)

    # initialize seed
    utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'])
    
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']

    supernet_train_chkpnt_fname = global_settings.NAS_EVOSEARCH_SETTINGS['TRAINED_SUPERNET_FNAME']
    supernet = get_supernet(global_settings, dataset, supernet_train_chkpnt_fname=supernet_train_chkpnt_fname)
    
    evo_search(global_settings, dataset, supernet)
