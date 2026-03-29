#adapted from: https://github.com/tinyalpha/mobileNet-v2_cifar10

import sys, os
import csv
import time
import logging
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from tensorboard import SummaryWriter


from .shuffle_arch import MNASSuperNet, MNASSubNet
from .common_utils import get_dataset

#sys.path.append("..")
from .. import utils
from .. import file_utils

sys.path.append("../..")
from settings import arg_parser, load_settings, Settings

from NASBase.model.common_utils import (
    blkchoices_ixs_to_blkchoices, blkchoices_to_blkchoices_ixs, get_network_dimension, get_network_obj, get_subnet, get_supernet, iter_blk_choices, netobj_to_pyobj, 
    get_sampled_subnet_configs
)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()







def validate(model, testloader, criterion, device):
    # validate model on testloader
    # return val_loss, val_acc
    
    model.eval()
    correct, total = 0, 0
    loss, counter = 0, 0
    
    with torch.no_grad():
        for (images, labels) in testloader:

            #images = images.cuda()
            #labels = labels.cuda()
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1
    
    return loss / counter, correct / total



# this is fully trian from scratch (not fine-tune)
def train(model: MNASSubNet, run_gpu_id, global_settings, input_resolution=None, num_data_loader_workers=8, 
          trained_supernet = None, # assume already loaded supernet
          fine_tune=False,
          dataset=None, train_epochs=None
          ):
    
    print("Train:Enter:: {}, {}".format(model.name, run_gpu_id))
    
    if dataset == None:
        dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    
    start_time = time.time()
    max_val_acc = 0
    
    train_loader, val_loader = get_dataset(global_settings, dataset=dataset, input_resolution=input_resolution, num_workers=num_data_loader_workers)
    
    #writer = SummaryWriter(log_dir='ts_logs')
 
    # write header
    file_utils.dir_create(global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'])
    logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR']  +  "subnet_" +  str(model.id) + "_" + model.name + '_trainlog.csv'
    
    with open(logfname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])
        
    #device_ids = run_gpu_id
    
    # build model and optimizer
    #model.cuda()
    #model = nn.DataParallel(net, device_ids = device_ids)
    
    
    
    # @TODO --- fine-tune [ fewer epochs than fully-training ]
    
    # == SUPERNET OPTION: 
    # if fine-tuning is selected, get the path of the subnet
    # train only that path (so during training a random path is not selected, select fixed path)
    # validation acc is reported only for that path
        
    # == SUBNET OPTION:
    # extract the weights from the supernet, corresponding to the subnet path
    # recreate a subnet py obj, initialize the weights using the extracted weights
    # continue training.   
    
    #subnet_path = []
    
    
    device = torch.device("cuda:"+str(run_gpu_id) if torch.cuda.is_available() else "cpu")
    print (device)
    #model.cuda()    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # train
    i = 0
    correct, total = 0, 0
    train_loss, counter = 0, 0
    
    train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_EPOCHS']    
    
    print("train_epochs = ", train_epochs)
    
    for epoch in range(train_epochs):
        epoch_start_time = time.time()

        # update lr
        #if epoch == 0:
        #    optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 4e-5, momentum = 0.9)
        #elif (epoch/train_epochs) >= 0.5:
        #    optimizer = optim.SGD(model.parameters(), lr = 1e-2, weight_decay = 4e-5, momentum = 0.9)
        #elif (epoch/train_epochs) >= 0.75:
        #    optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 4e-5, momentum = 0.9)

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_OPT_LR'], 
                                    momentum=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_MOM'], 
                                    weight_decay=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_WD']
                                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs)

        # iteration over all train data
        for data in train_loader:
            # shift to train mode
            model.train()
            
            # get the inputs
            inputs, labels = data
            #inputs = inputs.cuda()
            #labels = labels.cuda()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # count acc,loss on trainset
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()        
            train_loss += loss.item()
            counter += 1

            if i % 100 == 0:
                # get acc,loss on trainset
                acc = correct / total
                train_loss /= counter
                
                # validation
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                print('GPU[%d]: iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' 
                      %(run_gpu_id, i, epoch, train_loss, val_loss, acc, val_acc))
                
                # save logs and weights
                with open(logfname, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, train_loss, val_loss, acc, val_acc])
                
                if val_acc > max_val_acc:
                    #torch.save(model.state_dict(), 'weights.pkl') # <--- save trained weights
                    max_val_acc = val_acc
                    
                # reset counters
                correct, total = 0, 0
                train_loss, counter = 0, 0

            i += 1
        print("GPU[%d]: epoch time %.4f min" %(run_gpu_id, (time.time() - epoch_start_time)/60))

        # In trained_supernet.py, scheduler is called once in a epoch
        scheduler.step()
        
    return max_val_acc
