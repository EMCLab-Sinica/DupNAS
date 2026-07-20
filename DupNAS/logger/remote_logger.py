from datetime import datetime
import platform
import sys

RLogger = None

# -- wrapper class for a remote logger --
# may consider different remote logging service providers
class RemoteLogger:
    def __init__(self, proj_name, run_name, config, service_type='wandb', tags=[], notes="", group=""):
        self.proj_name = proj_name
        self.name = run_name
        self.config = config
        self.service_type = service_type
        self.tags = tags
        self.notes = notes
        self.group = group
        
        self._init_ok = False
        
    
    
    def init(self):        
        try:        
            if self.service_type == 'wandb':
                # start a new wandb run to track this script
                wandb.init(                
                    project = self.proj_name, # set the wandb project where this run will be logged
                    name = self.name,
                    config  = self.config,  # track hyperparameters and run metadata
                    tags    = self.tags, # used for filtering runs
                    notes   = self.notes,
                    group = self.group
                )
                self._init_ok = True
            else:
                sys.exit("RemoteLogger:: service not valid - {}".format(self.service_type))
        except Exception as e:
            print(e)
            
            
    def log(self, msg):
        if (msg != None) and (self._init_ok == True):
            if self.service_type == 'wandb':                
                wandb.log(msg)   # log metrics to wandb
            else:
                pass
                
    def save(self, filename):
        '''Specify the file to save. Only need to run once - files will be uploaded again whenever 
        '''
        if not self._init_ok:
            return

        if self.service_type == 'wandb':
            # Live saving overwrites uploaded files upon local changes
            wandb.save(filename, policy='live')
        else:
            pass
                
            
    def finish(self):
        if (self._init_ok == True):
            if self.service_type == 'wandb':                
                wandb.finish()  # [optional] finish the wandb run, necessary in notebooks
            else:
                pass
        
        

def get_remote_logger_basic_init_params(global_settings, run_name_suffix="", group_name_suffix=""):

    exp_suffix = global_settings.GLOBAL_SETTINGS['EXP_SUFFIX']
    init_params = {
        "rlog_proj_name" : global_settings.GLOBAL_SETTINGS['RLOGGER_PROJECT_NAME'],
        "rlog_run_name" : exp_suffix+(run_name_suffix or global_settings.GLOBAL_SETTINGS['REMOTE_LOGGER_RUN_NAME_SUFFIX']),
        "rlog_run_group" : exp_suffix+(group_name_suffix or global_settings.GLOBAL_SETTINGS['REMOTE_LOGGER_GROUP_NAME_SUFFIX']),
        "rlog_run_config" : {
            "argv" : sys.argv,
            "server"   : platform.node(),
            "exp_start" : datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "settings_obj" : global_settings.get_dict(),
        },
        "rlog_run_tags" : [exp_suffix],
    }

    return init_params

def get_remote_logger_obj(global_settings, rl_init_params=None, run_name_suffix=""):
    import wandb
    global RLogger

    # Already initialized, return the existing one
    if RLogger is not None:
        return RLogger

    # Initialization parameters are required for the first time
    assert rl_init_params

    if global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER']:
        RLogger = RemoteLogger(
            proj_name=rl_init_params['rlog_proj_name'],
            run_name=rl_init_params['rlog_run_name'],
            config=rl_init_params['rlog_run_config'],
            tags=rl_init_params["rlog_run_tags"],
            group=rl_init_params["rlog_run_group"]
        )
        RLogger.init()
        return RLogger
    else:
        return None
