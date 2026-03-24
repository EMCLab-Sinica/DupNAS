'''
Tune the evo search hyperparameters
'''


import os.path
import sys
from pprint import pprint

from settings import Settings, Stages, arg_parser, CURRENT_HOME_PATH
from NASBase import file_utils, utils
from NASBase.run_nas import run_nas

SEARCH_PARENT_RATIO = [0.1, 0.2, 0.3]
SEARCH_MUT_PROB = [0.1, 0.2, 0.3]
SEARCH_MUT_RATIO = [0.25, 0.5, 0.75]

#SEARCH_PARENT_RATIO = [0.1]   
#SEARCH_MUT_PROB = [0.1]
#SEARCH_MUT_RATIO = [0.25]



def main():
    global_settings = Settings() # default settings
    global_settings = arg_parser(global_settings)
    
    # -- common init
    global_settings.NAS_SETTINGS_GENERAL['STAGES'] = '3'
    
    # set the supernet to run the tuning
    global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME'] =  CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + 'TiNAS-M-threshold-0.1_ssoptlog.json',
    global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_TRAINED_SUPERNET_FNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + '_trsupnetresults.json'
        
    global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME'] = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME'][0]
    global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_TRAINED_SUPERNET_FNAME'] = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_TRAINED_SUPERNET_FNAME'][0]
    
    #pprint(global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME']); sys.exit()
    
    total_perms = len(SEARCH_PARENT_RATIO) * len(SEARCH_MUT_PROB) * len(SEARCH_MUT_RATIO)
    
    hyperparam_tuning_results = {}
    
    # evo param permutations
    ix = 1
    for evo_parent_ratio in SEARCH_PARENT_RATIO:
        for evo_mut_prob in SEARCH_MUT_PROB:
            for evo_mut_ratio in SEARCH_MUT_RATIO:
                                
                perm_k = "{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio)
                
                print("============= Running permutation [{}/{}]: {}, {}, {}".format(ix, total_perms, evo_parent_ratio, evo_mut_prob, evo_mut_ratio))
                
                global_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] = "evo_test_hyperparams_tune_{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio)
                global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + \
                                                                                global_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + "_resultlog.json",   
                                                                                
                global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'] = global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'][0]
                
    
                run_nas(global_settings) # this runs only evo search
                
                # save results
                stage_evo_search_logfname = global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'] 
                best_solution = file_utils.json_load(stage_evo_search_logfname)
                
                hyperparam_tuning_results[perm_k] = best_solution
                
                ix+=1
                
    
    hyperparam_tuning_results_fname = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + "evo_test_hyperparams_tune_allsummary_resultlog.json",   
    if isinstance(hyperparam_tuning_results_fname, tuple):
        hyperparam_tuning_results_fname = hyperparam_tuning_results_fname[0]
        
    file_utils.json_dump(hyperparam_tuning_results_fname, hyperparam_tuning_results)



if __name__ == '__main__':
    main()
