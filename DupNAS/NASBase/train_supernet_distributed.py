'''
Adapted from : https://github.com/ShunLu91/Single-Path-One-Shot-NAS
'''

import argparse
import logging
import os, sys
import time
import math
from pprint import pprint
import signal
import random
import multiprocessing
import datetime

import torch
import torch.nn as nn
import torch.multiprocessing
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import datasets
from torchinfo import summary

from NASBase import utils
#from NASBase.model.mnas_arch import MNASSuperNet

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

from NASBase.model.common_utils import get_dataset_multigpu
#from NASBase.model.common_utils import get_raw_dataset
from logger.remote_logger import get_remote_logger_obj, get_remote_logger_basic_init_params


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


trained_architectures = {}
    



# train loop per epoch
# fine_tune_subnet_blkchoices_ixs: for training only the specified subnet (if not None)
def train(rank, world_size, device, global_settings : Settings, tot_epochs, cur_epoch, train_loader, model: MNASSuperNet, criterion, optimizer,
          mode_txt, fine_tune_subnet_blkchoices_ixs=None):
    
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    try:
        num_choices_per_block = model.module.blk_choices #model.choices #model.module.choices
    except AttributeError:
        num_choices_per_block = model.blk_choices
    print("num_choices_per_block: ", len(num_choices_per_block))
    # print("----------------")
    # print("network choices:")
    # pprint(num_choices_per_block)
    # print("----------------")
    
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']
    
    print_freq = global_settings.NAS_SETTINGS_GENERAL['TRAIN_PRINT_FREQ']

    gradient_accumulation_steps = global_settings.NAS_SETTINGS_PER_DATASET[dataset].get('GRADIENT_ACCUMULATION_STEPS', 1)
    
    # Use random.sample to make sure every epoch skips the same number of batches
    batch_skip_ratio = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['BATCH_SKIP_RATIO']
    num_batches = len(train_loader)
    batches_to_skip = random.sample(range(num_batches), int(num_batches * batch_skip_ratio))
    last_print = 0
    for step, (inputs, targets) in enumerate(train_loader):
        if step in batches_to_skip:
            continue

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        if (fine_tune_subnet_blkchoices_ixs==None):
            if rank == 0 or world_size == 1:
                choices = utils.random_choice(len(num_choices_per_block), num_blocks)
            else:
                choices = [None] * num_blocks
            if world_size > 1:
                # Synchronize choices, so that batches in all processes are applied on the same subnet
                # https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast_object_list
                # https://github.com/pytorch/pytorch/issues/56142
                torch.distributed.broadcast_object_list(choices, src=0)
        else:
            choices = fine_tune_subnet_blkchoices_ixs
        
        #print(model._debug_get_tot_num_layers(choices))
        
        #print("-- choices: ", choices)
        outputs = model(inputs, choices)
        loss = criterion(outputs, targets)
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)
        if (step - last_print >= print_freq) or (step == (len(train_loader) - 1)):
            last_print = step
            logging.info(
                '[%s Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                % (mode_txt, lr, cur_epoch+1, tot_epochs, step+1, steps_per_epoch,
                   loss.item(), train_loss.avg, prec1, train_acc.avg)
            )
    return train_loss.avg, train_acc.avg
    

# validate loop per epoch
def validate(rank, world_size, device, global_settings : Settings, val_loader, model, criterion,
             fine_tune_subnet_blkchoices_ixs=None):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    try:
        num_choices_per_block = model.module.blk_choices #model.choices #model.module.choices
    except AttributeError:
        num_choices_per_block = model.blk_choices
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']
    
    max_prec1, min_prec1 = 0, 100

    with torch.no_grad():
        print('val_loader length:', len(val_loader)) # XXX debug more
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if (fine_tune_subnet_blkchoices_ixs==None):
                choices = utils.random_choice(len(num_choices_per_block), num_blocks)
            else:
                choices = fine_tune_subnet_blkchoices_ixs
                
            #print("-- choices: ", choices)
            outputs = model(inputs, choices)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)

            max_prec1 = max(max_prec1, prec1)
            min_prec1 = min(min_prec1, prec1)
    
    # report min and max val_acc
    if (fine_tune_subnet_blkchoices_ixs==None):
        logging.info('[Supernet Validation] max prec1: %.3f, min prec1: %.3f' % (max_prec1, min_prec1))
    else:
        logging.info('[Subnet Fine-Tune Validation] max prec1: %.3f, min prec1: %.3f' % (max_prec1, min_prec1))

    return val_loss.avg, val_acc.avg



def run_supernet_train(rank, world_size, queue, global_settings: Settings, dataset, trainset, valset, supernet_chkpt_fname, supernet,
                       fine_tune_subnet_blkchoices_ixs=None, train_epochs=None):
    if world_size > 1:
        signal.signal(signal.SIGUSR1, utils.debug)

    # create default process group
    if world_size > 1:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    
    print("run_supernet_train::Enter (fine_tune_subnet_blkchoices_ixs = {})".format(fine_tune_subnet_blkchoices_ixs))
        
    #logging.info(args)
    utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'])
     
    # -- Check Checkpoints Directory
    ckpt_dir = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR']
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    # -- Define Supernet 
    if dataset == None:
        dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    
    
    # -- create supernet --
    if supernet == None:
        test_out_ch_scale = 1.0    
        block_out_channels =  [math.ceil(test_out_ch_scale * c) for c in global_settings.NAS_SETTINGS_PER_DATASET[dataset]['OUT_CH_PER_BLK']]
        print("--- Generating the SuperNet")
        #print("block_out_channels = ", block_out_channels)        
        model = MNASSuperNet(global_settings, dataset, block_out_channels)
    else:
        model = supernet
    
    #summary(model, depth=2, input_size=(1, 3, 32, 32))    
    #logging.info(model)
    
    if (fine_tune_subnet_blkchoices_ixs==None):
        train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUPERNET_EPOCHS']
        lr = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_OPT_LR']
        trainset_batchsize = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_BATCHSIZE']
        mode_txt = "Supernet"
    else:
        train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_SUBNET_EPOCHS']
        lr = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_OPT_LR']
        trainset_batchsize = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_BATCHSIZE']
        mode_txt = "Subnet Fine-Tune"

    if world_size > 1:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size)
        valid_sampler = DistributedSampler(valset, num_replicas=world_size)
    else:
        train_sampler = torch.utils.data.RandomSampler(trainset)
        valid_sampler = torch.utils.data.RandomSampler(valset)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=trainset_batchsize, sampler=train_sampler,
                                               shuffle=False, pin_memory=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=global_settings.NAS_SETTINGS_PER_DATASET[dataset]['VAL_BATCHSIZE'], sampler=valid_sampler,
                                             shuffle=False, pin_memory=True, num_workers=0)

    ddp_kwargs = {}
    if len(model.blk_choices) > 1 and len(model.choice_blocks[0]) > 1:
        # In a supernet, each batch involves only some parameters in loss calculation, and DDP does not like that by default
        ddp_kwargs['find_unused_parameters'] = True

    model = model.to(device)
    if world_size > 1:
        # construct DDP model
        # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], **ddp_kwargs)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr,
                                momentum=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_MOM'], 
                                weight_decay=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_WD']
                                )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs)
    print('\n')

    rlog = None
    if rank == 0:
        if global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER']:
            rl_init_params = get_remote_logger_basic_init_params(global_settings, run_name_suffix=global_settings.GLOBAL_SETTINGS['REMOTE_LOGGER_RUN_NAME_SUFFIX'] + "_worker0")
            rl_init_params['rlog_run_tags'].extend(global_settings.GLOBAL_SETTINGS['REMOTE_LOGGER_EXTRA_TAGS'])
            rlog = get_remote_logger_obj(global_settings, rl_init_params)

    if rlog and supernet_chkpt_fname:
        rlog.save(supernet_chkpt_fname)

    val_loss, val_acc = validate(rank, world_size, device, global_settings, val_loader, model, criterion,
                                 fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
    logging.info(
        '[%s Validation] Before training val_loss: %.3f, val_acc: %.3f'
        % (mode_txt, val_loss, val_acc)
    )

    print("=== Starting Main Training Loop ===")

    # -- Training main loop
    start = time.time()
    best_val_acc = 0.0
    for epoch in range(train_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        # Supernet Training
        train_loss, train_acc = train(rank, world_size, device, global_settings, train_epochs, epoch, train_loader, model, criterion, optimizer, mode_txt,
                                      fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
        scheduler.step()
        logging.info(
            '[%s Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (mode_txt, epoch + 1, train_loss, train_acc)
        )
        
        # Supernet Validation
        val_loss, val_acc = validate(rank, world_size, device, global_settings, val_loader, model, criterion,
                                     fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)

        if world_size > 1:
            val_metrics_tensor = torch.tensor([val_loss, val_acc]).to(device)
            torch.distributed.all_reduce(val_metrics_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = val_metrics_tensor[0].item() / world_size
            avg_val_acc = val_metrics_tensor[1].item() / world_size
            logging.info('Average across GPUs: val_loss: %.3f, val_acc: %.3f' % (avg_val_loss, avg_val_acc))
        else:
            avg_val_acc = val_acc
            avg_val_loss = val_loss

        assert not math.isnan(avg_val_loss)

        # Save Best Supernet Weights
        if best_val_acc < avg_val_acc:
            best_val_acc = avg_val_acc
            best_val_loss = avg_val_loss            
            if supernet_chkpt_fname is not None:
                if rank == 0:
                    if not global_settings.NAS_SETTINGS_GENERAL['SEARCH_TIME_TESTING']:
                        supernet_chkpt_fname_with_timestamp = supernet_chkpt_fname.replace('.pth', str(datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')) + '.pth')

                        if (fine_tune_subnet_blkchoices_ixs==None):
                            torch.save({
                                'model': model.state_dict(),
                                'settings': str(global_settings),
                            }, supernet_chkpt_fname)
                            logging.info('Save best checkpoints to %s' % supernet_chkpt_fname)
                        else:
                            torch.save({
                                'model': model.state_dict(),
                                'settings': str(global_settings),
                            }, supernet_chkpt_fname_with_timestamp)
                            logging.info('Save best checkpoints to %s' % supernet_chkpt_fname_with_timestamp)
                    else:
                        logging.info('Testing search time, skipping saving checkpoints')
                else:
                    # The supernet is synchronized and identical across all proceses
                    logging.info('Not the first process, skipping saving the supernet')
            else:
                logging.warning('Model checkpoint filename is not specified, so the best checkpoint cannot be saved')
        logging.info(
            '[%s Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
            % (mode_txt, epoch + 1, val_loss, val_acc, best_val_acc)
        )

        if rlog:
            rlog.log({
                'mode': mode_txt,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
            })

        print('\n')

    ret = supernet_chkpt_fname, best_val_acc, best_val_loss
    if queue is not None:
        if rank == 0:
            queue.put(ret)
    else:
        return ret


def run_supernet_train_distributed(global_settings: Settings, dataset, supernet_chkpt_fname, supernet,
                       fine_tune_subnet_blkchoices_ixs=None, train_epochs=None):
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    # Not using random.randint here, as it results in the same number due to the given seed
    os.environ["MASTER_PORT"] = str(int(time.time() * 1e6) % 10000 + 30000)

    # -- get dataset
    _, input_resolution = supernet.net_choices
    #train_transform, _ = utils.data_transforms(dataset, input_resolution=input_resolution)
    
                                       # This transform is used for both train and val sets)

    world_size = torch.cuda.device_count()
    if world_size > 1:
        # the start method should match pytorch to avoid segmentation fault
        # https://discuss.pytorch.org/t/segfault-with-multiprocessing-queue/81292
        trainset, valset = get_dataset_multigpu(global_settings, dataset, input_resolution=input_resolution)
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()

        # https://pytorch.org/docs/stable/multiprocessing.html
        torch.multiprocessing.spawn(run_supernet_train,
            args=(world_size, queue, global_settings, dataset, trainset, valset, supernet_chkpt_fname, supernet,
                           fine_tune_subnet_blkchoices_ixs, train_epochs),
            nprocs=world_size,
            join=True)

        results = []
        while not queue.empty():
            results.append(queue.get())  # Get results from the queue

        return results[0]  # XXX: merge results
    else:
        return run_supernet_train(0, world_size, None, global_settings, dataset, trainset, valset, supernet_chkpt_fname, supernet,
                           fine_tune_subnet_blkchoices_ixs, train_epochs)




if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #'0,1'

    test_settings = Settings() # default settings
    test_settings = arg_parser(test_settings)
    run_supernet_train(test_settings)
    
        
    

