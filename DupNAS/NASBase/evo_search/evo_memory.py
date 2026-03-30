'''
Simple caching mechanism for EvoSearch
'''

import os, sys
import pprint
import enum

from NASBase.evo_search.utils import sample_blk_choice_str


class EvoMemTypes(enum.Enum):
    LAT     = 1
    ACC     = 2 # [val_loss, val_acc]
    NVM    = 3
    VM     = 4
    
    NVM_FIT = 5 
    VM_FIT = 6
     
    MAC = 7
    ACCESS = 8
    
    P_MAC = 9
    P_ACCESS = 10
    TS = 11
    

# Cahce is a dictionary, where keys identify subnets using a string format of subnet block structure,
# and in each cache entry there are multiple values (see EvoMemTypes)
class EvoMem:
    def __init__(self, global_settings_evosearch, net_choices, input_ch):
        self.evomem_enable = global_settings_evosearch['EVOSEARCH_ENABLE_EVOMEMORY']
        
        print("EvoMem:: ENABLE = ", self.evomem_enable)
        
        self.net_choices = net_choices
        self.input_ch = input_ch
        self.sample_tacking_tbl = {} # tracks different calculated features of samples: e.g., latency, accuracy etc.
        self.querystats = {'queries':0, 'hits':0} # [how many queries were made, how many were successful]

    # --- helpers ---    
    def _get_sample_key(self, sample):
        sn_blk_choice_key = sample_blk_choice_str(sample)           
        return sn_blk_choice_key
            
    
    def update_tbl_multival(self, sample, val_type_lst, val_lst):        
        if (self.evomem_enable):
            d = dict(zip(val_type_lst, val_lst))
            sn_key = self._get_sample_key(sample)
            
            if sn_key in self.sample_tacking_tbl:
                self.sample_tacking_tbl[sn_key].update(d)            
            else: # new entry
                self.sample_tacking_tbl[sn_key] = d
        else:
            pass
        
    
    def update_tbl(self, sample, val_type, val):
        if (self.evomem_enable):
            sn_key = self._get_sample_key(sample)
            
            if sn_key in self.sample_tacking_tbl:
                self.sample_tacking_tbl[sn_key].update({ val_type : val })            
            else: # new entry
                self.sample_tacking_tbl[sn_key] = { val_type : val }
        else:
            pass
   
    def query_tbl(self, sample, val_type):            
        if (self.evomem_enable):
            #self.querystats['queries']+=1     # doesn't work with multiprocessing   
            sn_key = self._get_sample_key(sample)
            
            if sn_key in self.sample_tacking_tbl:
                if val_type in self.sample_tacking_tbl[sn_key]:
                    #self.querystats['hits']+=1     # doesn't work with multiprocessing
                    return self.sample_tacking_tbl[sn_key][val_type]
                else:
                    return None
            else:
                return None
        else:
            return None
        
        
    def clean_tbl(self):
        if (self.evomem_enable):        
            self.sample_tacking_tbl = {}
        else:
            pass
    
    
    # def report_querystats(self):
    #     if self.querystats['queries']>0:
    #         print("EvoMem:: querystats = {}/{}={}".format(self.querystats['hits'], 
    #                                                   self.querystats['queries'],
    #                                                   (self.querystats['hits']/self.querystats['queries'])*100))
    #     else:
    #         print("EvoMem:: querystats = {}/{}=inf%".format(self.querystats['hits'], 
    #                                                         self.querystats['queries']
    #                                                         ))
        
    
