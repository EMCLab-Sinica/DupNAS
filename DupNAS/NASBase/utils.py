import csv
from datetime import datetime
import faulthandler
import logging
import math
import os
import random
import signal
import time
import traceback

#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms(dataset, input_resolution=None, cutout=False):
    if dataset == 'CIFAR10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        
        if input_resolution is None:
            input_resolution = (32, 32)

        train_transform = transforms.Compose([transforms.Resize(input_resolution),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])
    
        valid_transform = transforms.Compose([transforms.Resize(input_resolution),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        
        if cutout:
            cutout_len = 16
            train_transform.transforms.append(Cutout(cutout_len))

    elif dataset == 'IMAGE100':

        if input_resolution is None:
            input_resolution = 64

        train_transform = transforms.Compose([
            transforms.Resize((input_resolution, input_resolution)),         # Resize to 256x256
            #transforms.RandomCrop(input_resolution),     # Random crop to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((input_resolution,input_resolution)),
            #transforms.CenterCrop(input_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    else:
        raise ValueError("utils:dataset:: Error - unknown dataset : " + str(dataset))
    
    
    return train_transform, valid_transform


def random_choice(num_choice, num_blocks):
    return list(np.random.randint(num_choice, size=num_blocks))

# From https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates/53551417#53551417
#
# Return a randomized "range" using a Linear Congruential Generator
# to produce the number sequence. Parameters are the same as for 
# python builtin "range".
#   Memory  -- storage for 8 integers, regardless of parameters.
#   Compute -- at most 2*"maximum" steps required to generate sequence.
#
def random_range(start, stop=None, step=None):
    # Set a default values the same way "range" does.
    if (stop == None): start, stop = 0, start
    if (step == None): step = 1
    # Use a mapping to convert a standard range into the desired range.
    mapping = lambda i: (i*step) + start
    # Compute the number of numbers in this range.
    maximum = (stop - start) // step
    # Seed range with a random integer.
    value = random.randint(0,maximum)
    # 
    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    # 
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    # 
    offset = random.randint(0,maximum) * 2 + 1      # Pick a random odd-valued offset.
    multiplier = 4*(maximum//4) + 1                 # Pick a multiplier 1 greater than a multiple of 4.
    modulus = int(2**math.ceil(math.log2(maximum))) # Pick a modulus just big enough to generate all numbers (power of 2).
    # Track how many random numbers have been returned.
    found = 0
    while found < maximum:
        # If this is a valid value, yield it in generator fashion.
        if value < maximum:
            found += 1
            yield mapping(value)
        # Calculate the next value in the sequence.
        value = (value*multiplier + offset) % modulus

def random_choices(num_choice, num_blocks):
    num_net_choice = num_choice ** num_blocks

    generator = random_range(start=0, stop=num_net_choice)

    while True:
        subnet_choice_encoded = next(generator)
        block_choices = []
        for block_idx in range(num_blocks):
            subnet_choice_encoded, block_choice = divmod(subnet_choice_encoded, num_choice)
            block_choices.append(block_choice)
        yield tuple(block_choices)


# def plot_hist(acc_list, min=0, max=101, interval=5, name='search'):
#     plt.hist(acc_list, bins=max - min, range=(min, max), histtype='bar')
#     plt.xticks(np.arange(min, max, interval))
#     img_path = name + '.png'
#     plt.savefig(img_path)
#     plt.show()


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))

class CsvLogger:
    def __init__(self, filename, fields):
        self.output = open(filename, 'a')
        self.fields = fields
        self.writer = csv.writer(self.output)

        # Write the header if this is a newly-created file
        if self.output.tell() == 0:
            self.writer.writerow(['time'] + fields)

    def log(self, data):
        row = [
            datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        ]

        for item in data:
            row.append(item)

        self.writer.writerow(row)

        self.output.flush()

    def __del__(self):
        self.output.close()

def convergence_generation(arr):
    last = 0
    for idx, item in enumerate(arr):
        if item == arr[-1]:
            last = idx
            break
    return last

def count_changes(arr):
    #print(arr)
    changes = 0
    last = arr[0]
    for item in arr[1:]:
        if item != last:
            changes += 1
        last = item
    #print(changes)
    return changes


def enable_debug():
    faulthandler.enable()
    signal.signal(signal.SIGUSR1, debug)

# Print the current stack trace on demand to identify performance bottlenecks
# https://stackoverflow.com/questions/132058/showing-the-stack-trace-from-a-running-python-application
def debug(sig, frame):
    traceback.print_stack(frame)
