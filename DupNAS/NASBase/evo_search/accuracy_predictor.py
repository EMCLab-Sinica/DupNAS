'''
Predict subnet accuracy, using either:
- pre-trained supernet [DEFAULT]
- pre-constructed look-up-table [NOT IMPLEMENTED]
- estimation DNN model [NOT IMPLEMENTED]
'''


import os, sys
import torch.nn as nn
import torch
from pprint import pprint
import copy
import random

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchinfo import summary


#sys.path.append("../../..")
from NASBase import utils as utils
from NASBase import file_utils as file_utils


from NASBase.model.common_utils import blkchoices_to_blkchoices_ixs, get_dataset

from NASBase.evo_search.utils import sample_blk_choice_str



class AccuracyPredictor:
    def __init__(self, global_settings, trained_supernet, acc_tbl_fname=None):
        
        self.global_settings = global_settings
        self.supernet = trained_supernet
        
        if acc_tbl_fname != None:
            self.acc_table = self.load_acc_table(acc_tbl_fname)
        else:
            self.acc_table = None

    
    def load_acc_table(self, fname):        
        acc_data = file_utils.json_load(fname)    
        return acc_data
    
    
    
    def predict_accuracy(self, population, worker_id, input_resolution, cached_accs=None): 
        print("predict_accuracy:: WORKERID [%d] :: Enter : has %d jobs " % (worker_id, len(population)))
                         
        # -- if we are using a trained supernet to get subnet accuracy, cmmon initialize
        if (self.supernet != None) and (self.acc_table == None):
            device = torch.device("cuda:"+str(worker_id) if torch.cuda.is_available() else "cpu")                
            print (device)    
            torch.set_num_threads(1)
            
            # init cuda
            criterion = nn.CrossEntropyLoss().to(device)
            model = self.supernet    
            model = model.to(device)    
            model.eval()
            
            # get dataset
            train_loader, val_loader = get_dataset(self.global_settings, input_resolution=input_resolution, num_workers=0)
            
            # get all choices per blk
            supernet_choice_per_blk = self.supernet.blk_choices
            
        
        pop_accuracy = []
        for sample in population:  
            
            # check and query cached accuracies
            k = sample_blk_choice_str(sample)             
            if (cached_accs != None) and (k in cached_accs):
                val_acc = cached_accs[k]
            
            # calculate sample accuracy
            else:         
                if (self.acc_table):
                    # -- look up accuracy from table
                    _, val_acc = self.get_subnet_accuracy_tbl_lookup(sample)
                else:
                    # -- look up accuracy from supernet
                    if not isinstance(sample, list):
                        # sample may be a list or a numpy array, while blkchoices_to_blkchoices_ixs works with lists only
                        sample = sample.tolist()
                    subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_choice_per_blk, sample)       
                    _, val_acc = self.get_subnet_accuracy_from_supernet(device, model, criterion, subnet_choice_per_blk_ixs, train_loader, val_loader)
            
            #print("predict_accuracy:: WORKERID [%d] :: config=%s acc=%f" % (worker_id, str(subnet_choice_per_blk_ixs).replace(' ', ''), val_acc))
            pop_accuracy.append(val_acc)         
                
        return pop_accuracy
        
    
    
    
    def get_subnet_accuracy_from_supernet(self, device, model, criterion, subnet_choice_per_block, train_loader, val_loader):        
        # init    
        val_loss = utils.AverageMeter()
        val_acc = utils.AverageMeter()
        
        val_loss.reset(); val_acc.reset()
        
        with torch.no_grad(): # inference only
            for step, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, subnet_choice_per_block)
                loss = criterion(outputs, targets)
                prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                n = inputs.size(0)
                val_loss.update(loss.item(), n)
                val_acc.update(prec1.item(), n)
                
        return val_loss.avg, val_acc.avg
        
    
    def get_subnet_accuracy_tbl_lookup(self, subnet_config_choice_per_blk):    
        sb_blk_choice_key = "<" + ','.join([str(c) for c in subnet_config_choice_per_blk]) + ">"        
        acc = random.randint(50,100) #DEBUGGING - use random accuracy
        #acc = self.acc_table[sb_blk_choice_key]    
        #print(sb_blk_choice_key, acc); sys.exit()
        return acc

        
        
  
