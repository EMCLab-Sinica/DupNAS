import functools
import multiprocessing
import os
import random

import torch
from NASBase.utils import set_seed

def get_max_num_workers(worker_type):
    if worker_type == 'CPU':
        max_num_workers = multiprocessing.cpu_count()
    elif worker_type == 'GPU':
        max_num_workers = torch.cuda.device_count()
    else:
        raise ValueError('Invalid worker type: {}'.format(worker_type))

    return max_num_workers

def available_gpus():
    return list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))

def worker_func_wrapper(worker_func, random_seed, *args, **kwargs):
    set_seed(random_seed)
    return worker_func(*args, **kwargs)

# https://stackoverflow.com/questions/56562962/how-can-i-pass-two-arguments-to-pool-starmap
def run_multiprocessing_workers(num_workers, worker_func, worker_type, common_args, worker_args):
    '''
    This helper function runs a given function on multiple workers with specified common and
    worker-specific arguments. Results from workers are returned as a list, where each element
    is the result of the worker function.

    - num_workers (int): The number of workers.
    - worker_func (function): The function to run on individual workers. It will get several
                              arguments. The first argument is the device index. Following
                              arguments are common arguments (common_args) and then worker-specific
                              arguments (worker_args).
    - worker_type (string): The worker type. It should be either one of "CPU" or "GPU".
    - common_args (list): Common arguments for all workers. Each list element will be passed as an
                          argument to the worker function.
    - worker_args (list): Worker-specific arguments. Its length
    '''

    # Checking arguments
    max_num_workers = get_max_num_workers(worker_type)
    if num_workers > max_num_workers:
        raise ValueError('The number of workers ({}) exceeds the number available devices with type {} ({})'.format(
                         num_workers, worker_type, max_num_workers))

    if len(worker_args) not in (0, num_workers):
        raise ValueError('The number of elements in worker_args ({}) is not the same as the number of workers ({}).'.format(len(worker_args), num_workers))


    # Combine arguments for workers
    args_lists = []

    # Set deterministic and worker-dependent seeds
    worker_random_seed_base = random.randint(0, 2**32-1)
    args_lists.append([
        worker_random_seed_base + n
        for n in range(num_workers)
    ])

    if worker_type == 'GPU' and 'CUDA_VISIBLE_DEVICES' in os.environ:
        args_lists.append(available_gpus())
    else:
        args_lists.append(range(num_workers))

    for one_common_arg in common_args:
        args_lists.append([one_common_arg] * num_workers)

    if worker_args:
        args_lists.append(worker_args)

    combined_args = zip(*args_lists)


    # Run workers
    if True:
        ctx = multiprocessing.get_context('spawn')  # CUDA requires to use 'spawn' start method

        with ctx.Pool(num_workers) as p:
            results = p.starmap(functools.partial(worker_func_wrapper, worker_func), combined_args)
    else:
        # Skip multiprocessing to make debugging easier
        results = [functools.partial(worker_func_wrapper, worker_func)(*arg_list) for arg_list in combined_args]

    return results
