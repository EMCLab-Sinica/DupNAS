# ========================= NASBase.py =========================
# (Only Stage 2 runs multi-GPU when --dist!=none)
# =============================================================

'''
Stage 1: SS optimization
Stage 2: Evo search
'''

import sys
import os.path
import datetime
import re
from .train_supernet import run_supernet_train
#from .train_supernet_distributed import run_supernet_train_distributed

from settings import Settings, SSOptPolicy, Stages, arg_parser
from NASBase import file_utils, utils
from NASBase.evo_search.search import evo_search
from NASBase.model.common_utils import get_supernet, parametric_supernet_choices, parametric_supernet_blk_choices
from NASBase.ss_optimization.ss_opt import ss_optimization
from NASBase.fine_tune import fine_tune_best_solution
from logger.remote_logger import get_remote_logger_obj, get_remote_logger_basic_init_params



# ----------------- Helpers for Stage-2-only distributed -----------------
# [ADDED] helpers to read torchrun env (no global dist init)
def get_env_rank() -> int:  # [ADDED]
    import os
    return int(os.environ.get("RANK", "0"))

def get_env_world_size() -> int:  # [ADDED]
    import os
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_rank0_env() -> bool:  # [ADDED]
    return get_env_rank() == 0
# ------------------------------------------------------------------------


def stage_ss_optimization(global_settings: Settings, dataset, supernet_name, stage_ss_opt_logfname):
    print("-- stage_ss_optimization::Enter")

    # ========= STAGE 1: optimize search space ========
    supernet_choices, _ = parametric_supernet_choices(global_settings=global_settings)
    # block-level choices are not dropped in stage 1, so global_settings are not passed to parametric_supernet_blk_choices
    supernet_block_choices = parametric_supernet_blk_choices(global_settings=global_settings)

    best_supernet_config, supernet_properties = ss_optimization(
        global_settings,
        dataset,
        supernet_choices,           # Level 1 search space
        supernet_block_choices,     # Level 2 search space
    )

    stage_ss_opt_results = {
        'best_supernet_config': best_supernet_config,
        'best_supernet_blk_choices': None,  # The default will be used if this is None
        'num_subnets': supernet_properties['num_subnets'],
        'supernet_type': supernet_properties['supernet_objtype'],
        'dataset': dataset,
        'supernet_name': supernet_name,
        #'trained_supernet_fname': "{}/test_supernet_{}_oneshot_train_best.pth".format(global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR'], 
        #                                                                              supernet_properties['supernet_objtype'])            
        
        # this specifies where to save the trained supernet
        'trained_supernet_fname' : "{}_supernet_{}_best.pth".format(
                                                                    global_settings.GLOBAL_SETTINGS['EXP_SUFFIX'],
                                                                    supernet_properties['supernet_objtype']
                                                                    ),
        'per_supernet_stats': supernet_properties['per_supernet_stats'],
    }

    if global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_POLICY'] == SSOptPolicy.FLOPS:
        stage_ss_opt_results['average_flops'] = supernet_properties['average_flops']

    # save stage SS_OPT results to JSON (dataset, supernet type, suitable configurations for the supernet)
    # overwrite json
    file_utils.delete_file(stage_ss_opt_logfname)
    file_utils.json_dump(stage_ss_opt_logfname, stage_ss_opt_results)

def create_supernet(global_settings: Settings, dataset, stage_ss_opt_logfname,
                    load_state=False):
    print("create_supernet::Enter")
    
    # if trained supernet is provided, then open that file. corresponding supernet config also should be provided
    # if (stage_ss_opt_logfname != None) and (stage_ss_opt_logfname != ""):
        
    #     stage_ss_opt_results = file_utils.json_load(stage_ss_opt_logfname)
    #     supernet_train_chkpnt_fname = trained_supernet_fname
        
    #     print("trained_supernet_fname : ", trained_supernet_fname)
    #     print("trained_supernet_config : ", trained_supernet_config)
        
    #     width_multiplier, input_resolution = trained_supernet_config
        
    # else: 
    
    # load stage SS_OPT data from JSON. Use settings in the JSON to create the supernet    
    print("stage_ss_opt_logfname =", repr(stage_ss_opt_logfname))
    stage_ss_opt_results = file_utils.json_load(stage_ss_opt_logfname)
    best_supernet_config = stage_ss_opt_results['best_supernet_config']
    width_multiplier, input_resolution = best_supernet_config
    print(f"Using supernet config {best_supernet_config}")
    blk_choices = stage_ss_opt_results['best_supernet_blk_choices']        

    original_supernet_train_chkpnt_fname = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR'] + stage_ss_opt_results['trained_supernet_fname']
    if global_settings.NAS_TESTING_SETTINGS['TRAINED_SUPERNET_FNAME']:
        supernet_train_chkpnt_fname = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR'] + global_settings.NAS_TESTING_SETTINGS['TRAINED_SUPERNET_FNAME']
    else:
        supernet_train_chkpnt_fname = original_supernet_train_chkpnt_fname
    
    print("--------------")
    print(stage_ss_opt_logfname)
    print(best_supernet_config)
    print(supernet_train_chkpnt_fname)
    print("--------------")
    
    
    #stage_train_supernet_results = file_utils.json_load(stage_train_supernet_logfname)        
    #supernet_train_chkpnt_fname = os.path.basename(stage_train_supernet_results['supernet_best_ckpt'])
        

    supernet = get_supernet(global_settings, dataset,
                            load_state=load_state, supernet_train_chkpnt_fname=supernet_train_chkpnt_fname,
                            width_multiplier=width_multiplier, input_resolution=input_resolution,
                            blk_choices=blk_choices)

    return supernet, supernet_train_chkpnt_fname, original_supernet_train_chkpnt_fname



def stage_train_supernet(global_settings: Settings, dataset, supernet_name, supernet, stage_train_supernet_logfname, supernet_train_chkpnt_fname):
    print("-- stage_train_supernet::Enter")

    # --- train supernet (one-shot) ---
    if not global_settings.NAS_SETTINGS_PER_DATASET[dataset].get('TRAIN_MULTI_GPU', False):
        supernet_best_ckpt, best_val_acc, best_val_loss = run_supernet_train(global_settings, dataset, supernet_chkpt_fname=supernet_train_chkpnt_fname, supernet=supernet)
    else:
        supernet_best_ckpt, best_val_acc, best_val_loss = run_supernet_train_distributed(global_settings, dataset, supernet_chkpt_fname=supernet_train_chkpnt_fname, supernet=supernet)

    # stage_train_supernet_results = {
    #     'supernet_best_ckpt': supernet_best_ckpt,
    #     'best_val_acc': best_val_acc,
    #     'best_val_loss': best_val_loss,
        
        # should we sample 1000 rnd subnets, and add more info - like : 
        # best subnet acc
        # worst subnet acc
        # avg subnet acc
    # }

    # save stage TRAIN_SUPERNET results to JSON
    # overwrite json

    if is_rank0_env():
        stage_train_supernet_results = {
            'supernet_best_ckpt': supernet_best_ckpt,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
        }

        file_utils.delete_file(stage_train_supernet_logfname)
        file_utils.json_dump(stage_train_supernet_logfname, stage_train_supernet_results)

        print(
            f"Stage 2 result saved to {stage_train_supernet_logfname}: "
            f"best_val_acc={best_val_acc}, "
            f"best_val_loss={best_val_loss}",
            flush=True,
        )

    # stage_train_supernet_logfname_with_timestamp = stage_train_supernet_logfname.replace('.json', str(datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')) + '.json')
    # file_utils.delete_file(stage_train_supernet_logfname)
    # file_utils.json_dump(stage_train_supernet_logfname, stage_train_supernet_results)


def stage_evo_search(global_settings, dataset, best_supernet, stage_evo_search_logfname):
    print("-- stage_evo_search::Enter")

    # -- run evo search
    # Run N times with different seeds and get the best out of N
    N = global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_TRIALS']
    exp_suffix = global_settings.GLOBAL_SETTINGS['EXP_SUFFIX']

    best_acc_all_trials = 0
    best_solution_all_trials = None

    #print(best_supernet)

    for run_id in range(N):
        utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'] + run_id)

        best_solution = evo_search(global_settings, dataset, best_supernet, stage_evo_search_logfname, run_id=run_id, exp_suffix=f'{exp_suffix}-{run_id}')

         # compare best_solution in each file to find the best
        _, best_info, _ = best_solution
        best_acc = best_info['accuracy']

        if best_acc > best_acc_all_trials:
            best_solution_all_trials = best_solution
            best_acc_all_trials = best_acc

    assert best_solution_all_trials

    # # save stage EVO_SEARCH results to JSON
    # # overwrite json
    file_utils.delete_file(stage_evo_search_logfname)
    file_utils.json_dump(stage_evo_search_logfname, best_solution_all_trials)


def stage_fine_tune(global_settings: Settings, dataset, supernet, supernet_chkpt_fname, stage_ss_opt_logfname, stage_evo_search_logfname, stage_fine_tune_logfname):
    fine_tune_base_generation = global_settings.NAS_TESTING_SETTINGS['FINETUNE_BASE_GENERATION']
    if fine_tune_base_generation is not None:
        run_id = 0 # assume only one run
        stage_evo_search_logfname = stage_evo_search_logfname.replace('.json', f'-{run_id}-gen{fine_tune_base_generation}.json')
        stage_fine_tune_logfname = stage_fine_tune_logfname.replace('.json', f'-{run_id}-gen{fine_tune_base_generation}.json')

    stage_ss_opt_results = file_utils.json_load(stage_ss_opt_logfname)
    best_solution = file_utils.json_load(stage_evo_search_logfname)

    best_solution_info = fine_tune_best_solution(global_settings, dataset, supernet, supernet_chkpt_fname, stage_ss_opt_results['best_supernet_config'], best_solution)

    # overwrite json
    #file_utils.delete_file(stage_fine_tune_logfname)
    #file_utils.json_dump(stage_fine_tune_logfname, best_solution_info)
    if is_rank0_env():
        file_utils.delete_file(stage_fine_tune_logfname)
        file_utils.json_dump(
            stage_fine_tune_logfname,
            best_solution_info,
        )

        print(
            f"Stage 4 result saved to {stage_fine_tune_logfname}",
            flush=True,
        )


def initialize(global_settings: Settings):
    # randomizers
    utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'])

    # gpus select
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = global_settings.CUDA_SETTINGS['GPUIDS']
    
    # Remote logger
    # Not keeping the result, just initialize it
    if global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER'] and is_rank0_env():  
        rl_init_params = get_remote_logger_basic_init_params(global_settings)
        rl_init_params['rlog_run_tags'].extend(global_settings.GLOBAL_SETTINGS['REMOTE_LOGGER_EXTRA_TAGS'])
        get_remote_logger_obj(global_settings, rl_init_params)
    
# [ADDED] Stage-2-only dist init/cleanup
def _stage2_init_dist_if_needed(global_settings: Settings) -> bool:  # [ADDED]
    dist_mode = global_settings.GLOBAL_SETTINGS.get('DIST_MODE', 'none')
    if dist_mode in ('ddp', 'fsdp'):
        import os
        import torch
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True
    return False

def _stage2_cleanup_dist_if_needed(global_settings: Settings):
    # Do not call dist.destroy_process_group() here.
    # On some servers, NCCL shutdown hangs even after all ranks finish.
    pass
#-----------------------------------------------

def run_nas(global_settings: Settings):
    initialize(global_settings)
    stages_completed = []

    stages = list(map(int, global_settings.NAS_SETTINGS_GENERAL['STAGES'].split(',')))  # from a command line argument
    if stages == []:
        sys.exit('run_nas:: Error - stages empty')
        
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']

    train_log_dir = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR']
    exp_suffix = global_settings.GLOBAL_SETTINGS['EXP_SUFFIX']

    # ---- EXP SETUP 
    file_utils.dir_create(train_log_dir)

    #base_name = re.sub(r'-\d+$', '', exp_suffix)
    #supernet_name = base_name + '_supernet'
    
    supernet_name = exp_suffix + '_supernet'
    
    stage_ss_opt_logfname = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME']
    stage_train_supernet_logfname = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_TRAINED_SUPERNET_FNAME']
    stage_evo_search_logfname = global_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'] 
    stage_fine_tune_logfname = train_log_dir + exp_suffix + '_best_solution_info.json'
        
    # -- STAGE 1: Search Space Optimization
    if Stages.SS_OPT in stages and is_rank0_env(): #(rank-0 only)  [MODIFIED]
        stage_ss_optimization(global_settings, dataset, supernet_name, stage_ss_opt_logfname)
        stages_completed.append(Stages.SS_OPT)

    # -- STAGE 2: Train Supernet (one-shot)
    if Stages.TRAIN_SUPERNET in stages:        
        used_dist = _stage2_init_dist_if_needed(global_settings)  # [ADDED] init only for Stage-2

        supernet, supernet_train_chkpnt_fname, _ = create_supernet(global_settings, dataset, stage_ss_opt_logfname, 
                                                                load_state=False)  
        
        if used_dist or is_rank0_env():  # [MODIFIED] run train on all ranks if dist
            stage_train_supernet(global_settings, dataset, supernet_name, supernet, stage_train_supernet_logfname, supernet_train_chkpnt_fname)

        _stage2_cleanup_dist_if_needed(global_settings)  # [ADDED] tear down      
        #stage_train_supernet(global_settings, dataset, supernet_name, supernet, stage_train_supernet_logfname, supernet_train_chkpnt_fname)
        stages_completed.append(Stages.TRAIN_SUPERNET)

    # -- STAGE 3: Search Supernet (evolutionary search)
    if Stages.EVO_SEARCH in stages and is_rank0_env(): #(rank-0 only)  [MODIFIED]
        best_supernet, supernet_train_chkpnt_fname, _ = create_supernet(global_settings, dataset, stage_ss_opt_logfname, 
                                                                    load_state=True)        
        stage_evo_search(global_settings, dataset, best_supernet, stage_evo_search_logfname)            
        stages_completed.append(Stages.EVO_SEARCH)

    # -- STAGE 4: Fine tune the best solution
    if Stages.FINE_TUNE in stages:
        used_dist = _stage2_init_dist_if_needed(global_settings)
        best_supernet, supernet_train_chkpnt_fname, original_supernet_train_chkpnt_fname = create_supernet(global_settings, dataset, stage_ss_opt_logfname, 
                                                                                                       load_state=True)
        if used_dist or is_rank0_env():  #[MODIFIED] run train on all ranks if dist
            stage_fine_tune(global_settings, dataset, best_supernet, original_supernet_train_chkpnt_fname,
                        stage_ss_opt_logfname, stage_evo_search_logfname, stage_fine_tune_logfname)
        


        _stage2_cleanup_dist_if_needed(global_settings)  # [ADDED] tear down  


        if is_rank0_env():
            stages_completed.append(Stages.FINE_TUNE)


def main():
    test_settings = Settings() # default settings
    test_settings = arg_parser(test_settings)
    run_nas(test_settings)
    


if __name__ == '__main__':
    main()

    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)