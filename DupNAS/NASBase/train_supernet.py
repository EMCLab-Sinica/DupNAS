'''
Adapted from : https://github.com/ShunLu91/Single-Path-One-Shot-NAS
'''

import argparse
import logging
import os, sys
import time
import math
from pprint import pprint
# [ADDED]
import contextlib
import datetime
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchinfo import summary
import torch.distributed as dist  # <-- [ADDED] THIS

from NASBase import utils

#sys.path.append("..")
from settings import Settings, arg_parser, load_settings
from NASBase.model.common_utils import get_dataset
from logger.remote_logger import get_remote_logger_obj



if Settings.NAS_SETTINGS_GENERAL['ARC'] == 'mbv2':
    from NASBase.model.mbv2_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.mbv2_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'shuffle':
    from NASBase.model.shuffle_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.shuffle_ss import *
elif Settings.NAS_SETTINGS_GENERAL['ARC'] == 'incept':
    from NASBase.model.inception_arch import MNASSuperNet, MNASSubNet
    from NASBase.model.inception_ss import *

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


trained_architectures = {}
    

# [ADDED] helpers for distributed env
def _rank() -> int:
    return int(os.environ.get("RANK", "0"))

def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def _use_dist(settings: Settings) -> bool:
    return _world_size() > 1 and settings.GLOBAL_SETTINGS.get('DIST_MODE', 'none') in ('ddp','fsdp')

# unwrap DDP/FSDP to access original module attributes
def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def _sync_choices(choices, device):
    if not _is_dist():
        return choices
    t = torch.tensor(choices, dtype=torch.int64, device=device)
    dist.broadcast(t, src=0)
    return t.tolist()


# train loop per epoch
# fine_tune_subnet_blkchoices_ixs: for training only the specified subnet (if not None)
def train(device, global_settings : Settings, tot_epochs, cur_epoch, train_loader, model: MNASSuperNet, criterion, optimizer,
          mode_txt, fine_tune_subnet_blkchoices_ixs=None):
    
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    #num_choices_per_block = model.blk_choices #model.choices #model.module.choices
    
    # [MODIFIED]
    if global_settings.GLOBAL_SETTINGS['DIST_MODE'] == 'none':
        num_choices_per_block = model.blk_choices #model.choices #model.module.choices
    else:
        m = _unwrap_model(model)
        num_choices_per_block = m.blk_choices
    

    if _is_rank0():  # [ADDED] avoid 4x noisy prints
        print("num_choices_per_block: ", len(num_choices_per_block))
    
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']
    
    print_freq = global_settings.NAS_SETTINGS_GENERAL['TRAIN_PRINT_FREQ']
    gradient_accumulation_steps = global_settings.NAS_SETTINGS_PER_DATASET[dataset].get('GRADIENT_ACCUMULATION_STEPS', 1)
    
    # Use random.sample to make sure every epoch skips the same number of batches
    batch_skip_ratio = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['BATCH_SKIP_RATIO']
    num_batches = len(train_loader)
    batches_to_skip = random.sample(range(num_batches), int(num_batches * batch_skip_ratio))
    last_print = 0
    
    for step, (inputs, targets) in enumerate(train_loader):
        # [MODIFIED] move to the correct device for this rank
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        if (fine_tune_subnet_blkchoices_ixs==None):
            #randomize choices each step, sync across ranks
            choices = utils.random_choice(len(num_choices_per_block), num_blocks)
        else:
            #choices = fine_tune_subnet_blkchoices_ixs
            if _is_dist():
                if _is_rank0():
                    choices = fine_tune_subnet_blkchoices_ixs #utils.random_choice(len(m.blk_choices), num_blocks)
                else:
                    choices = [0] * num_blocks  # placeholder
                choices = _sync_choices(choices, device)
            else:
                #choices = utils.random_choice(len(num_choices_per_block), num_blocks)
                choices = fine_tune_subnet_blkchoices_ixs

        outputs = model(inputs, choices)
        loss = criterion(outputs, targets)
        loss.backward()
        assert not torch.isnan(loss)
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)

        # [MODIFIED] only rank-0 prints (or last step to see something if single-rank)
        should_log = _is_rank0()
        if (step % print_freq == 0 or step == (len(train_loader) - 1)) and should_log:
            logging.info(
                '[%s Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                % (mode_txt, lr, cur_epoch+1, tot_epochs, step+1, steps_per_epoch,
                   loss.item(), train_loss.avg, prec1, train_acc.avg)
            )
    return train_loss.avg, train_acc.avg
    

# validate loop per epoch
def validate(device, global_settings : Settings, val_loader, model, criterion,
             fine_tune_subnet_blkchoices_ixs=None):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']

    #num_choices_per_block = model.blk_choices
    
    # [MODIFIED]
    if global_settings.GLOBAL_SETTINGS['DIST_MODE'] == 'none':
        num_choices_per_block = model.blk_choices #model.choices #model.module.choices
    else:
        m = _unwrap_model(model)
        num_choices_per_block = m.blk_choices
    
    max_prec1, min_prec1 = 0, 100

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            # [MODIFIED]
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if (fine_tune_subnet_blkchoices_ixs==None):
                #randomize choices each step, sync across ranks
                choices = utils.random_choice(len(num_choices_per_block), num_blocks)
            else:
                #choices = fine_tune_subnet_blkchoices_ixs
                if _is_dist():
                    if _is_rank0():
                        choices = fine_tune_subnet_blkchoices_ixs  #utils.random_choice(len(m.blk_choices), num_blocks)
                    else:
                        choices = [0] * num_blocks  # placeholder
                    choices = _sync_choices(choices, device)
                else:
                    #choices = utils.random_choice(len(m.blk_choices), num_blocks)
                    choices = fine_tune_subnet_blkchoices_ixs
                    
            outputs = model(inputs, choices)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)

            max_prec1 = max(max_prec1, prec1)
            min_prec1 = min(min_prec1, prec1)
    
    # [MODIFIED] only rank-0 logs min/max
    if _is_rank0():
        if (fine_tune_subnet_blkchoices_ixs==None):
            logging.info('[Supernet Validation] max prec1: %.3f, min prec1: %.3f' % (max_prec1, min_prec1))
        else:
            logging.info('[Subnet Fine-Tune Validation] max prec1: %.3f, min prec1: %.3f' % (max_prec1, min_prec1))

    return val_loss.avg, val_acc.avg



def run_supernet_train(global_settings: Settings, dataset=None, supernet_chkpt_fname=None, supernet=None,
                       fine_tune_subnet_blkchoices_ixs=None, train_epochs=None):

    # [MODIFIED] device per rank (fixes all 4 ranks fighting for cuda:0)
    local_rank = _local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else torch.device("cpu"))
    
    print("run_supernet_train::Enter (fine_tune_subnet_blkchoices_ixs = {})".format(fine_tune_subnet_blkchoices_ixs))
    
    utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'])

    # -- Check Checkpoints Directory
    ckpt_dir = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR']
    if _is_rank0() and (not os.path.exists(ckpt_dir)):
        os.mkdir(ckpt_dir)

    # [ADD] DDP process-group init (safe)
    init_pg = False
    if _use_dist(global_settings) and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        init_pg = True
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    try:
        # -- Define Supernet 
        if dataset == None:
            dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
        
        # -- create supernet --
        if supernet == None:
            test_out_ch_scale = 1.0    
            block_out_channels =  [math.ceil(test_out_ch_scale * c) for c in global_settings.NAS_SETTINGS_PER_DATASET[dataset]['OUT_CH_PER_BLK']]
            if _is_rank0():
                print("--- Generating the SuperNet")
            model = use_dist (global_settings, dataset, block_out_channels)
        else:
            model = supernet
        
        #summary(model, depth=2, input_size=(1, 3, 32, 32))    
        
        # [MODIFIED] epochs/lr/batch from settings as before
        if (fine_tune_subnet_blkchoices_ixs==None):
            train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUPERNET_EPOCHS']
            lr = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_OPT_LR']
            base_batch = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_BATCHSIZE']
            mode_txt = "Supernet"
        else:
            train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_SUBNET_EPOCHS']
            lr = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_OPT_LR']
            base_batch = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_BATCHSIZE']
            mode_txt = "Subnet Fine-Tune"

        # [ADDED] per-GPU or global batch handling
        per_gpu_bs = base_batch
        if global_settings.GLOBAL_SETTINGS.get('BATCH_SCOPE', 'per_gpu') == 'global' and _world_size() > 1:
            per_gpu_bs = max(1, base_batch // _world_size())

        # -- get dataset (returns DataLoaders); we will rebuild with DistributedSampler if needed
        _, input_resolution = model.net_choices
        orig_train_loader, orig_val_loader = get_dataset(global_settings, input_resolution=input_resolution, trainset_batchsize=per_gpu_bs)

        # [ADDED] rebuild loaders with DistributedSampler under DDP/FSDP
        from torch.utils.data import DataLoader, DistributedSampler
        use_dist = _use_dist(global_settings)

        if use_dist:
            train_sampler = DistributedSampler(orig_train_loader.dataset)
            val_sampler   = DistributedSampler(orig_val_loader.dataset, shuffle=False)
            train_loader = DataLoader(orig_train_loader.dataset, batch_size=per_gpu_bs, sampler=train_sampler,
                                      shuffle=False, num_workers=4, pin_memory=True)
            val_loader   = DataLoader(orig_val_loader.dataset, batch_size=per_gpu_bs, sampler=val_sampler,
                                      shuffle=False, num_workers=4, pin_memory=True)
        else:
            train_sampler = None
            val_sampler = None
            train_loader = orig_train_loader
            val_loader = orig_val_loader

        # [MODIFIED] model/device + optional DDP/FSDP wrap and AMP
        dist_mode = global_settings.GLOBAL_SETTINGS.get('DIST_MODE', 'none')
        amp_mode  = global_settings.GLOBAL_SETTINGS.get('AMP', 'off')

        model = model.to(device)

        if dist_mode == "ddp" and use_dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False) # broadcast_buffers=False : important for per-GPU different subnets
        elif dist_mode == "fsdp" and use_dist:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
            mp = None
            if amp_mode == "fp16":
                mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
            elif amp_mode == "bf16":
                mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
            model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mp)

        # loss/opt/sched (as before)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_MOM'], 
                                    weight_decay=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_WD']
                                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs)
        print('\n') if _is_rank0() else None

        # [MODIFIED] remote logger only on rank-0 to avoid duplicates
        rlog = None
        if _is_rank0() and global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER']:
            rlog = get_remote_logger_obj(global_settings)

        if _is_rank0() and rlog and supernet_chkpt_fname:
            rlog.save(supernet_chkpt_fname)

        # [ADDED] AMP contexts
        scaler = torch.cuda.amp.GradScaler() if amp_mode == "fp16" else None
        if amp_mode in ("fp16","bf16"):
            autocast_ctx = torch.cuda.amp.autocast
        else:
            autocast_ctx = contextlib.nullcontext

        # initial val
        val_loss, val_acc = validate(device, global_settings, val_loader, model, criterion,
                                     fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
        if _is_rank0():
            logging.info(
                '[%s Validation] Before training val_loss: %.3f, val_acc: %.3f'
                % (mode_txt, val_loss, val_acc)
            )

        if _is_rank0():
            print("=== Starting Main Training Loop ===")

        # -- Training main loop
        start = time.time()
        best_val_acc = 0.0
        best_val_loss = float("inf")   # [ADDED] init in case no improvement

        for epoch in range(train_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # train
            if scaler:
                # with AMP autocast inside train loop
                # (we’ll keep train() simple and rely on autocast here)
                pass

            # [MODIFIED] training with optional AMP
            model.train()
            train_acc = utils.AverageMeter()
            train_loss = utils.AverageMeter()
            steps_per_epoch = len(train_loader)

            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()

                # random subnet / fixed subnet
                #num_choices_per_block = model.module.blk_choices if hasattr(model, "module") else model.blk_choices
                
                # [MODIFIED]
                if global_settings.GLOBAL_SETTINGS['DIST_MODE'] == 'none':
                    num_choices_per_block = model.blk_choices #model.choices #model.module.choices
                else:
                    m = _unwrap_model(model)
                    num_choices_per_block = m.blk_choices
                

                num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']

                if (fine_tune_subnet_blkchoices_ixs==None):
                    #randomize choices each step, sync across ranks
                    choices = utils.random_choice(len(num_choices_per_block), num_blocks)
                else:
                    #choices = fine_tune_subnet_blkchoices_ixs
                    if _is_dist():
                        if _is_rank0():
                            choices = fine_tune_subnet_blkchoices_ixs  #utils.random_choice(len(m.blk_choices), num_blocks)
                        else:
                            choices = [0] * num_blocks  # placeholder
                        choices = _sync_choices(choices, device)
                    else:
                        #choices = utils.random_choice(len(num_choices_per_block), num_blocks)
                        choices = fine_tune_subnet_blkchoices_ixs

                if amp_mode in ("fp16","bf16"):
                    with autocast_ctx():
                        outputs = model(inputs, choices)
                        loss = criterion(outputs, targets)
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    outputs = model(inputs, choices)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                assert not torch.isnan(loss)
                prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                n = inputs.size(0)
                train_loss.update(loss.item(), n)
                train_acc.update(prec1.item(), n)

                if (_is_rank0() and
                    (step % global_settings.NAS_SETTINGS_GENERAL['TRAIN_PRINT_FREQ'] == 0 or step == (len(train_loader) - 1))):
                    logging.info('[%s Training] epoch: %03d/%03d step: %03d/%03d train_loss: %.3f(%.3f) train_acc: %.3f(%.3f)'
                                 % (mode_txt, epoch+1, train_epochs, step+1, steps_per_epoch,
                                    loss.item(), train_loss.avg, prec1, train_acc.avg))

            scheduler.step()
            if _is_rank0():
                logging.info('[%s Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
                             (mode_txt, epoch + 1, train_loss.avg, train_acc.avg))
            
            # validate
            val_loss, val_acc = validate(device, global_settings, val_loader, model, criterion,
                                         fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
            
            # Save Best Supernet Weights (rank-0 only)
            if _is_rank0() and best_val_acc < val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss            
                if supernet_chkpt_fname is not None:
                    # [MODIFIED] handle DDP/FSDP state dict
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    supernet_chkpt_fname_with_timestamp = supernet_chkpt_fname.replace('.pth', str(datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')) + '.pth')
                    if (fine_tune_subnet_blkchoices_ixs==None):
                        torch.save(state_dict, supernet_chkpt_fname)
                    else:
                        torch.save(state_dict, supernet_chkpt_fname_with_timestamp)
                    
                    logging.info('Save best checkpoints to %s' % supernet_chkpt_fname_with_timestamp)
                else:
                    logging.warning('Model checkpoint filename is not specified, so the best checkpoint cannot be saved')

            if _is_rank0():
                logging.info('[%s Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
                             % (mode_txt, epoch + 1, val_loss, val_acc, best_val_acc))

            if _is_rank0() and (global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER']):
                rlog = get_remote_logger_obj(global_settings)
                rlog.log({
                    'mode': mode_txt,
                    'epoch': epoch + 1,
                    'train_loss': train_loss.avg,
                    'train_acc': train_acc.avg,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                })

            if _use_dist(global_settings):
                # [ADDED] keep ranks in sync each epoch
                torch.distributed.barrier()

            if _is_rank0():
                print('\n')

        # [ADDED] final barrier for clean exit when distributed
        if _use_dist(global_settings):
            torch.distributed.barrier()

        return supernet_chkpt_fname, best_val_acc, best_val_loss

    finally:
        # [ADD] tear down PG so torchrun can exit cleanly
        if init_pg and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #'0,1'

    test_settings = Settings() # default settings
    test_settings = arg_parser(test_settings)
    run_supernet_train(test_settings)
