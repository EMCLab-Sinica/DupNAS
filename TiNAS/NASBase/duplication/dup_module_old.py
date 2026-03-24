# === Revision: Micro-Graph Splitting Integrated ===
import onnx
import os
import pprint
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pprint import PrettyPrinter
from itertools import permutations, product
from datetime import datetime
from collections import defaultdict, deque
import time

from settings import Settings
# === Parse command-line arguments ===
# parser = argparse.ArgumentParser(description="Select memory optimization mode")
# parser.add_argument("--mode", type=str, default="none", choices=["pdq", "tinyts", "tinynas", "none"],
#                     help="Choose one of: pdq, tinyts, tinynas, none (none for tracing memory usage only)")
# parser.add_argument("--priority", type=str, default="mem", choices=["mem", "bal"],
#                     help="Memory optimization goal priority: mem or bal")
# parser.add_argument("--export_file", action='store_true',
#                     help="Enable exporting reports and figures")
# parser.add_argument("--vmsize", type=int, default=512,
#                     help="Set memory constraint in KB (e.g., 32 for 32KB)")
# parser.add_argument("--onnx", type=str, default='sample_0',
#                     help="Choose an input onnx model")

# args = parser.parse_args()
# # Mode flags
# mode = args.mode

mode = Settings.NAS_SETTINGS_GENERAL['MODE']
GOAL_PRIORITY = Settings.NAS_SETTINGS_GENERAL['GOAL']
ENABLE_PDQSEL_MODE = mode == 'pdq'
ENABLE_MICROGRAPH_MODE = mode == 'tinyts'
ENABLE_TINYNAS_MODE = mode == 'tinynas'

# Other settings
EXPORT_FILE = Settings.NAS_SETTINGS_GENERAL['EXP_FILE']
#GOAL_PRIORITY = 'bal'
VMSIZE = Settings.NAS_SETTINGS_GENERAL['VMSIZE']  # Convert KB to bytes
AVAILABLE_VM = int(VMSIZE) * 1024
DIR_PATH = "./"
#MODEL_NAME = args.onnx

VALID_Q = list(range(2,33,1))
SP_HEIGHT_LIST = [2,1]
#pprint = pprint.PrettyPrinter().pprint
inplace_op=['Relu', 'Softmax','MaxPool', 'GlobalAveragePool', 'Squeeze', 'Add', 'Mul', 'Reshape',  'BatchNormalization', 'Sigmoid', 'Dropout','LRN', 'DequantizeLinear', 'QuantizeLinear', 'LeakyRelu','Split', 'Slice']
unsplittable_ops = ['Gemm', 'MatMul', 'GlobalAveragePool','Softmax','Flatten', 'Reshape','Transpose','Split', 'Gather' , 'Slice', 'Sub', 'Div']  # You can change this as needed
merging_ops = ['Sum', 'Add','Concat']  # You can change this as needed

unsplittable_op_indices = []
# max_MAC_gap = 0.0

#Q_LIST = [2,4,8,16,32]
pp = PrettyPrinter(width=150)


# === [NEW] Micrograph splitter ===
# def split_into_micrographs(node_info, unsplittable_ops, data_usage):
#     micrographs = []
#     current_micro = []
#     unsplittable_op_indices.clear()  # Reset list on each use

#     for idx, node in enumerate(node_info):

#         if len(data_usage[node['data_out'][0]]['to']) > 1:
#             if current_micro:
#                 current_micro.append(idx)
#                 micrographs.append(current_micro)
#                 current_micro = []
#             continue
#         if node['op_type'] in unsplittable_ops:
#             if current_micro:
#                 micrographs.append(current_micro)
#                 current_micro = []
#             unsplittable_op_indices.append(idx)  # Track instead of isolating
#             continue
#         current_micro.append(idx)

#     if current_micro:
#         micrographs.append(current_micro)

#     return micrographs

# def split_into_micrographs(node_info, unsplittable_ops, data_usage):
#     """
#     Split the model into micrographs with these rules:
#       1) Unsplittable ops are hard boundaries and are NOT included in any micrograph.
#       2) If a node's output fans out (branch root), create one micrograph PER outgoing branch.
#          Each branch micrograph starts with the branch root, then follows exactly one path.
#       3) Stop a branch micrograph BEFORE the next boundary:
#            - before an unsplittable op, or
#            - before a join op (node with multiple inputs), or
#            - before the next split (fan-out > 1) to keep "one-branch-only".
#       4) Linear regions without branching are grouped as before, but never cross boundaries.
#     Returns: List[List[int]] of node indices (each micrograph is a list of node indices).
#     """

#     N = len(node_info)

#     def successors(idx):
#         """All consumer node indices of ANY output tensor of node idx."""
#         succ = set()
#         for t in node_info[idx].get('data_out', []):
#             for j in data_usage[t].get('to', []):
#                 succ.add(j)
#         return sorted(succ)

#     def is_unsplittable(idx):
#         return node_info[idx]['op_type'] in unsplittable_ops

#     def is_merging(idx):
#         return node_info[idx]['op_type'] in merging_ops

#     def is_join(idx):
#         # A 'join' is any op that consumes multiple inputs (e.g., Concat/Add/…).
#         # We look at tensor fan-in via number of input tensors.
#         # (If your graph packs extra meta, adapt this check accordingly.)
#         return len(node_info[idx].get('data_in', [])) > 1

#     def is_split(idx):
#         # A 'split' is any op whose output goes to multiple successors.
#         return len(successors(idx)) > 1

#     micrographs = []
#     visited_linear = set()  # to avoid duplicating linear segments when no branching

#     # ---- 1) Handle explicit branching: fan-out nodes spawn per-branch micrographs
#     for i in range(N):
#         if is_unsplittable(i) or is_join(i):
#             continue  # boundaries themselves are excluded
#         succs = successors(i)
#         if len(succs) <= 1:
#             continue  # not a split root

#         # For each outgoing branch, walk forward along exactly one path
#         for s in succs:
#             path = [i]  # include the branch root in every branch micrograph
#             curr = s

#             # Walk until the next boundary **before** stepping onto it
#             while True:
#                 # If the current node is a boundary, stop BEFORE adding it
#                 if is_unsplittable(curr) or is_join(curr) or is_split(curr):
#                     break

#                 # Safe to include
#                 path.append(curr)

#                 # Step forward
#                 next_succs = successors(curr)
#                 if len(next_succs) != 1:
#                     # 0 => end of graph; >1 => would introduce a new split (not allowed)
#                     break
#                 curr = next_succs[0]

#             if len(path) > 1:  # must contain at least [root, one more] to be meaningful
#                 micrographs.append(path)

#             # Mark nodes from linear walks so we don't repeat them later
#             for n in path:
#                 visited_linear.add(n)

#     # ---- 2) Handle remaining linear regions (no branching)
#     # We sweep through nodes and aggregate straight chains between boundaries,
#     # but do not cross into nodes already covered in branch paths.
#     i = 0
#     while i < N:
#         if i in visited_linear or is_unsplittable(i) or is_join(i):
#             i += 1
#             continue

#         # Start a new linear micrograph only if this node isn't already in a branch micrograph
#         # and it's not a split (splits were handled above)
#         if is_split(i):
#             i += 1
#             continue

#         path = []
#         curr = i
#         while True:
#             if curr in visited_linear or is_unsplittable(curr) or is_join(curr) or is_split(curr):
#                 break
#             path.append(curr)
#             visited_linear.add(curr)
#             next_succs = successors(curr)
#             if len(next_succs) != 1:
#                 break
#             curr = next_succs[0]

#         if path:
#             micrographs.append(path)

#         # Advance i past this linear chunk (or just move one step if empty)
#         i = max(i + 1, (path[-1] + 1) if path else i + 1)

#     return micrographs

def split_into_micrographs(node_info, unsplittable_ops, merging_ops, data_usage):
    """
    Split the model into micrographs:
      - Multi-branch (fan-out > 1): one micrograph per branch, excluding shared & merge nodes.
      - Single path (no fan-out/fan-in): one micrograph covering the linear chain between boundaries.
    
    Boundaries:
      - unsplittable op
      - merge/join op (fanin > 1 or op_type in merging_ops)
      - split (fanout > 1, treated specially at roots)
    """
    N = len(node_info)

    def successors(idx):
        succ = set()
        for t in node_info[idx].get('data_out', []):
            for j in data_usage[t].get('to', []):
                succ.add(j)
        return sorted(succ)

    def fanin(idx): return len(node_info[idx].get('data_in', []))
    def fanout(idx): return len(successors(idx))

    def is_unsplittable(idx): return node_info[idx]['op_type'] in unsplittable_ops
    def is_merge(idx): return (node_info[idx]['op_type'] in merging_ops) or (fanin(idx) > 1)
    def is_split(idx): return fanout(idx) > 1

    micrographs = []
    visited = set()

    # ---- Case A: multibranch
    for root in range(N):
        if is_unsplittable(root) or is_merge(root): 
            continue
        if not is_split(root): 
            continue

        for s in successors(root):
            path = []
            curr = s
            while True:
                if is_unsplittable(curr) or is_merge(curr) or is_split(curr):
                    break
                path.append(curr)
                visited.add(curr)
                nxt = successors(curr)
                if len(nxt) != 1: break
                curr = nxt[0]
            if path:
                micrographs.append(path)

    # ---- Case B: leftover linear paths
    for i in range(N):
        if i in visited: continue
        if is_unsplittable(i) or is_merge(i) or is_split(i): 
            continue

        path = []
        curr = i
        while True:
            if curr in visited or is_unsplittable(curr) or is_merge(curr) or is_split(curr):
                break
            path.append(curr)
            visited.add(curr)
            nxt = successors(curr)
            if len(nxt) != 1: break
            curr = nxt[0]

        if path:
            micrographs.append(path)

    return micrographs


# === NEW: Tensor-splitting & duplication per micrograph ===
def estimate_operator_duplication(micro_ops, node_info, split_height=1):
    duplicated_ops = []
    split_tensors =[]
    for op_idx in micro_ops:
        node = node_info[op_idx]
        ishapes = node['input_shapes']
        oshapes = node['output_shapes']


        # Skip if shape info is missing or invalid
        if not ishapes or not ishapes[0] or len(ishapes[0]) < 3:
            continue  # skip unknown or malformed shape

        H = ishapes[0][2]  # input height
        Ho = oshapes[0][2]  # input height
        
        if not H or H < split_height:
            num_splits = 1
            split_ts_num = 1
        else:
            split_ts_num = Ho // split_height
            if node['op_type'] == 'Conv':
                num_splits = Ho // split_height
                #kh = node['attributes'].get('kernel_shape', [1, 1])[0]
                # kh = node['attributes']['kernel_shape'][0]
                # stride = node['attributes'].get('strides', [1, 1])[0]
                # pad_top = node['attributes'].get('pads', [0, 0, 0, 0])[0]
                # pad_bottom = node['attributes'].get('pads', [0, 0, 0, 0])[2]
                # H_padded = Ho + pad_top + pad_bottom
                # num_splits = max(1, (H_padded - kh) // stride + 1)
            else:
                num_splits = Ho // split_height

        # Simulate duplication count (for copy)
        if num_splits > 1:
            #print(f"  [Split] Op {op_idx} ({node['op_type']}) split into {num_splits} copies, ishapes:{ishapes}, oshapes:{oshapes}")
            duplicated_ops.append((op_idx, num_splits))
            split_tensors.append((op_idx, split_ts_num))

    return duplicated_ops, split_tensors

# === Build expanded ops and tensor lifetime for each micrograph ===
# def build_expanded_ops(micro_ops, duplicated_ops, split_tensors):
#     op_copies = {}
#     split_tensor_map = dict(split_tensors)
#     for (op_idx, num_copies) in duplicated_ops:
#         split_value = split_tensor_map.get(op_idx, num_copies)
#         op_copies[op_idx] = []
#         for i in range(split_value):
#             input_tensor = f"split_{op_idx}_{i}"
#             output_tensor = f"out_{op_idx}_{i}"
#             op_copies[op_idx].append((f"{op_idx}_{i}", input_tensor, output_tensor))
#     return op_copies

def build_expanded_ops(micro_ops, duplicated_ops, split_tensors, node_info):
    """
    For each op, create q copies and assign input/output tensors based on dependency.
    """
    op_copies = {}
    op_tensor_map = {}
    split_tensor_map = dict(split_tensors)

    for op_idx in micro_ops:

        if op_idx in split_tensor_map:
            split_value = split_tensor_map[op_idx]
            op_copies[op_idx] = []

            producer_idx = node_info[op_idx]["data_in"][0]
            out_idx = node_info[op_idx]["data_out"][0]

            
            for i in range(split_value):
                input_tensor = f"split_{producer_idx}_{i}"
                output_tensor = f"split_{out_idx}_{i}"
                op_copies[op_idx].append((f"{op_idx}_{i}", input_tensor, output_tensor, split_value))
        else:
            input_tensor = node_info[op_idx]['data_in'][0]
            output_tensor = node_info[op_idx]['data_out'][0]
            op_copies[op_idx] = [(f"{op_idx}_0", input_tensor, output_tensor, 1)]
#     for sidx, op_idx in enumerate(micro_ops):
#         _, split_value = split_tensors[sidx]
# #         
#         if any(op_idx == dup_idx for (dup_idx, _) in duplicated_ops):
#             op_copies[op_idx] = []

    
#             producer_idx = node_info[op_idx]["data_in"][0]
#             out_idx = node_info[op_idx]["data_out"][0]

#             for i in range(split_value):
#                 # input from previous op in graph
#                 #producer_op_idx = find_input_producer(op_idx, node_info)  # you need this
#                 input_tensor = f"split_{producer_idx}_{i}"
#                 output_tensor = f"split_{out_idx}_{i}"
#                 op_copies[op_idx].append((f"{op_idx}_{i}", input_tensor, output_tensor, split_value))
#         else:
#             # Not duplicated
#             input_tensor = node_info[op_idx]['data_in'][0]
#             output_tensor = node_info[op_idx]['data_out'][0]
#             op_copies[op_idx] = [(f"{op_idx}_0", input_tensor, output_tensor,1)]

    return op_copies

def schedule_ops(op_copies, mode='dfs'):
    scheduled = []
    if mode == 'dfs':
        # DFS: preserve original topological ordering, then copy index
        # e.g., for original order [op1, op2], do [op1_0, op2_0, op1_1, op2_1, ...]
        original_order = list(op_copies.keys())
        max_copies = max(len(copies) for copies in op_copies.values())
        for i in range(max_copies):
            for op_idx in original_order:
                if i < len(op_copies[op_idx]):
                    scheduled.append(op_copies[op_idx][i])
    else:
        # BFS: group all copies of the same operator
        for op_idx in sorted(op_copies):
            scheduled.extend(op_copies[op_idx])
    return scheduled

def get_split_tensor_lifetime(schedule, data_usage):
    """
    Compute lifetime for split tensors under dependency-preserving scheduling.
    schedule: list of (op_name, input_tensor, output_tensor)
    q: number of splits (e.g., 2, 4)
    """
    tensor_life = {}

    # Step 1: Build map from split tensor to ops
    #for s in range(len(schedule):
    tensor_producer = {}
    tensor_consumer = {}
    tensor_size = {}


    for step, (op_name, input_tensor, output_tensor, split_value) in enumerate(schedule):
        # Record producer
        if isinstance(input_tensor, str) and input_tensor.startswith("split_"):
            _, prod_idx, copies = input_tensor.split("_")
            tensor_idx = int(prod_idx)
            size = int(data_usage[tensor_idx]['size']) // int(split_value)
            if data_usage[tensor_idx]['from'] == []:
                generating_op = "input_" + str(copies)
            else:
                generating_op_idx = data_usage[tensor_idx]['from'][0]
                generating_op = str(generating_op_idx) + "_" + str(copies)          

            using_op_idx = data_usage[tensor_idx]['to'][-1]
            using_op = str(using_op_idx) + "_" + str(copies)
        else:
            tensor_idx = int(input_tensor)
            size = int(data_usage[tensor_idx]['size'])
            generating_op_idx = data_usage[tensor_idx]['from'][0]
            using_op_idx = data_usage[tensor_idx]['to'][-1]
            generating_op = str(generating_op_idx) 
            using_op = str(using_op_idx)



        #generating_op_ = data_usage[tensor_idx]['from'][0]
        #using_op = data_usage[tensor_idx]['to'][0]
        
        if input_tensor not in tensor_producer:
            tensor_producer[input_tensor] = generating_op
            tensor_size[input_tensor] = size

        if input_tensor not in tensor_consumer:
            tensor_consumer[input_tensor] = using_op
        # else: 
        #     if input_tensor.startswith("split_"):
        #         cons_idx, copies = tensor_consumer[input_tensor].split("_")
        #     else:
        #         cons_idx = tensor_consumer[input_tensor]
            
        #     if using_op_idx > int(cons_idx):
        #         tensor_consumer[input_tensor] = using_op
        #     print(f"using_op_idx:{using_op_idx}, cons_idx:{cons_idx}")
        #print(f"input_tensor: {input_tensor}, tensor_producer: {tensor_producer}\n")
        #print(f"input_tensor: {input_tensor}, tensor_consumer: {tensor_consumer}\n")

        if isinstance(output_tensor, str) and output_tensor.startswith("split_"):
            _, prod_idx, copies = output_tensor.split("_")
            tensor_idx = int(prod_idx)
            size = int(data_usage[tensor_idx]['size']) // int(split_value)
            generating_op_idx = data_usage[tensor_idx]['from'][0]
            using_op_idx = data_usage[tensor_idx]['to'][-1]
            generating_op = str(generating_op_idx) + "_" + str(copies)
            using_op = str(using_op_idx) + "_" + str(copies)
        else:
            tensor_idx = int(output_tensor)
            size = int(data_usage[tensor_idx]['size'])
            generating_op_idx = data_usage[tensor_idx]['from'][0]
            using_op_idx = data_usage[tensor_idx]['to'][-1]
            generating_op = str(generating_op_idx)
            using_op = str(using_op_idx)


        #generating_op = data_usage[output_tensor]['from'][0]
        #using_op = data_usage[output_tensor]['to'][0]
        
        if output_tensor not in tensor_producer:
            tensor_producer[output_tensor] = generating_op
            tensor_size[output_tensor] = size

        if output_tensor not in tensor_consumer:
            tensor_consumer[output_tensor] = using_op
        # else: 
        #     if isinstance(output_tensor, str) and output_tensor.startswith("split_"):
        #         cons_idx, copies = tensor_consumer[output_tensor].split("_")
        #     else:
        #         cons_idx, copies = tensor_consumer[output_tensor]

        #     if using_op_idx > int(cons_idx):
        #         tensor_consumer[output_tensor] = using_op
        #     print(f"using_op_idx:{using_op_idx}, cons_idx:{cons_idx}")
         #tensor_producer[input_tensor] = generating_op
        #tensor_consumer[input_tensor] = using_op
        # Record consumer
        #if input_tensor not in tensor_consumer:
         #   tensor_consumer[input_tensor] = []
        #tensor_consumer[input_tensor].append(step)

    # Step 2: Use [producer, max(consumer)] as lifetime
    #print(f" tensor_producer: {tensor_producer}\n")
    #print(f" tensor_consumer: {tensor_consumer}\n")
       
    all_tensors = list(tensor_producer.keys())
    for tid in all_tensors:
        start = tensor_producer[tid] #.get(tid, min(tensor_consumer[tid]))  # fallback
        end = tensor_consumer[tid]  #.get(tid, [start]))
        size = tensor_size[tid]
        tensor_life[tid]  = [start, end, size]

    return tensor_life


# def get_tensor_lifetime(schedule, op_copies, schedule_mode):
#     tensor_life = {}
#     output_buffers = {}
#     produced_outputs = []

#     for step, (op_name, input_tensor, output_tensor) in enumerate(schedule):
#         if input_tensor:
#             tid = str(input_tensor)
#             if tid not in tensor_life:
#                 tensor_life[tid] = [step, step]
#             else:
#                 tensor_life[tid][1] = step

#         produced_outputs.append(output_tensor)
#         if output_tensor not in tensor_life:
#             tensor_life[output_tensor] = [step, step]

#         if schedule_mode == 'dfs':
#             op_prefix = op_name.split('_')[0]
#             output_buffers.setdefault(op_prefix, []).append(output_tensor)
#             if len(output_buffers[op_prefix]) > 1:
#                 for t in output_buffers[op_prefix]:
#                     tensor_life[t][1] = step

#         elif schedule_mode == 'bfs':
#             parent_op = int(op_name.split('_')[0]) - 1
#             parent_prefix = f"{parent_op}_"
#             for t in list(tensor_life.keys()):
#                 if t.startswith("split_" + parent_prefix):
#                     tensor_life[t][1] = step

#     return tensor_life

def build_opname_to_step(schedule):
    """
    Create a mapping from op_name (e.g., "6_59") to its step index in the schedule.
    """
    #check num of copies 
    max_num = 0
    #opname_to_step_tmp = {}
    opname_to_step = {}
    for step, item in enumerate(schedule):
        op_name = item[0]  # Only care about first field
        opname_to_step[op_name] = step

        op, num = item[0].split("_")
        if op == "0":
            if int(num) > max_num:
                max_num = int(num)

    #print(f"max_num for input: {max_num}")   
    
    for opkey in opname_to_step:
        opname_to_step[opkey] += max_num

    for mi in range(max_num):
        name = "input_" + str(mi)
        opname_to_step[name] = mi
  
    #print(opname_to_step)
    return opname_to_step


def estimate_peak_memory_from_lifetime(tensor_life, data_usage, opname_to_step):
    """
    Estimate the peak memory required to store all live tensors based on lifetime intervals.
    """
    timeline = {}  # step index -> live memory in bytes

    for tname, (start_opname, end_opname, size) in tensor_life.items():
        if start_opname not in opname_to_step or end_opname not in opname_to_step:
            continue  # skip if we can't resolve op names

        start = opname_to_step[start_opname]
        end = opname_to_step[end_opname]

        # Parse tensor index and copies for size
        if tname.startswith("split_"):
            _, tidx, copies = tname.split("_")
            size = int(size)
        else:
            try:
                size = int(data_usage[int(tname)]["size"])
            except (ValueError, IndexError, TypeError):
                continue

        # Add memory usage across lifetime
        for step in range(start, end + 1):
            timeline[step] = timeline.get(step, 0) + size

    # Peak memory is max total live memory across all steps
    peak_mem = max(timeline.values()) if timeline else 0
    #print(f"    => Peak Memory: {peak_mem} bytes ({peak_mem // 1024} KB)")
    return peak_mem, timeline

def bin_packing_memory_allocation(tensor_life, data_usage, node_info, opname_to_step, set_of_op_start_br, set_of_op_end_br):
    """
    Estimate the peak memory required to store all live tensors based on lifetime intervals.
    """
    allocations = []  # list of (start_step, end_step, size, tensor_name)
    retain_from_br_in = 0
    retain_from_br_out = 0
    #print("set_of_op_start_br: ", set_of_op_start_br)
    for tname, (start_opname, end_opname, size) in tensor_life.items():
        #print(f"tname:{tname} - start_opname: {start_opname}, end_opname: {end_opname}, size: {size}")
        # if start_opname.startswith("input_"):
        #     start = opname_to_step[end_opname]
        #     end = opname_to_step[end_opname]

        # else:
        if retain_from_br_in == 0:
            from_op, copies = start_opname.split("_")
            #print("from_op: ", from_op)
            if start_opname.startswith("input"):
                retain_from_br_in = 0
            else:
                if int(from_op) in set_of_op_start_br:
                #print("start_opname, start_op :",start_opname, from_op)
                    retain_from_br_in = data_usage[node_info[int(from_op)]['data_out'][0]]['size']
                #print(f"retain_from_br_in: {retain_from_br_in}")


        if start_opname not in opname_to_step :
            continue  # skip if we can't resolve op names
        
        if end_opname not in opname_to_step:
            opname_to_step[end_opname] = len(opname_to_step)


        
        start = opname_to_step[start_opname]
        end = opname_to_step[end_opname]

        # Parse tensor index and copies for size
        if tname.startswith("split_"):
            _, tidx, copies = tname.split("_")
            size = int(size)
        else:
            try:
                size = int(data_usage[int(tname)]["size"])
            except (ValueError, IndexError, TypeError):
                continue

        allocations.append((start, end, size, tname))

        
    # === Sort tensors by start time (or larger size first if desired)
    #print(allocations)
    allocations.sort(key=lambda x: (x[0], x[1], -x[2]))  # (start_time, -size)
    #print(allocations)
    memory_blocks = []  # list of [offset, size, end_time]

    tensor_offsets = {}
    current_offset = 0

    for start, end, size, tname in allocations:
        assigned = False

        # Try to fit into existing blocks
        for idx, (offset, block_size, block_end) in enumerate(memory_blocks):
            if block_end < start and block_size >= size:
                # Reuse this block
                tensor_offsets[tname] = offset
                memory_blocks[idx] = (offset, size, end)
                assigned = True
                break

        if not assigned:
            # Place it at the end
            tensor_offsets[tname] = current_offset
            memory_blocks.append((current_offset, size, end))
            current_offset += size

    # Peak memory is max total live memory across all steps
    peak_memory = max(offset + size for offset, size, _ in memory_blocks) if memory_blocks else 0
 
    #print(f"    => Bin Packing Peak Memory: {peak_memory} bytes ({peak_memory // 1024} KB)")
    return peak_memory, tensor_offsets, retain_from_br_in



def conv_out_size_1d(L_in, pad_before, pad_after, k, stride):
    # No dilation
    eff_k = k
    
    return math.floor((L_in + pad_before + pad_after - eff_k) / stride) + 1

def estimate_conv_macs_split_H(
    H, W, C_in, C_out,
    kh, kw,
    stride_h=1, stride_w=1,
    pad_top=0, pad_left=0, pad_bottom=0, pad_right=0,
    q=1,
    patch_style_overlap=False,
):
    """
    Split along height into q tiles.

    patch_style_overlap = False:
        Ideal output-partition tiling (no duplicated outputs).
        MACs == base.

    patch_style_overlap = True:
        "Patching" like flows: each tile takes a local input with halo
        (overlap = kh - stride_h on internal seams), runs conv on that local input,
        producing *more* output rows than it keeps, then crops. Those extra rows are
        the duplicated computation overhead.

    Returns:
        base_macs, macs_split
    """
    # --- base output sizes ---
    H_out = conv_out_size_1d(H, pad_top, pad_bottom, kh, stride_h)
    W_out = conv_out_size_1d(W, pad_left, pad_right, kw, stride_w)

    macs_per_out = C_in * kh * kw
    base_macs = C_out * H_out * W_out * macs_per_out

    if q <= 1:
        return base_macs, base_macs

    # Partition output rows across q tiles as evenly as possible
    base_rows = H_out // q
    rem = H_out % q
    rows_per_tile = [(base_rows + 1 if i < rem else base_rows) for i in range(q)]

    if not patch_style_overlap:
        # Ideal case: compute exactly assigned outputs, once.
        macs_split = C_out * (sum(rows_per_tile)) * W_out * macs_per_out
        return base_macs, macs_split  # equals base_macs

    # --- patching-style overlap accounting ---
    # Input overlap per internal seam
    overlap = max(0, kh - stride_h)

    total_rows_effective = 0
    for i, assigned_rows in enumerate(rows_per_tile):
        # Halo for internal seams
        halo_top = overlap if i > 0 else 0
        halo_bot = overlap if i < q - 1 else 0

        # Local input height needed to produce 'assigned_rows' outputs:
        # assigned_rows = floor((L_local + pads_local - kh)/stride_h) + 1
        # Invert to a minimal local input that yields exactly 'assigned_rows':
        # L_local_min = (assigned_rows - 1)*stride_h + kh
        # Now add halos to let neighbors' receptive fields fit in this tile's local input
        L_local = (assigned_rows - 1) * stride_h + kh + halo_top + halo_bot

        # Local vertical padding only exists at global top/bottom
        pad_local_top = pad_top if i == 0 else 0
        pad_local_bot = pad_bottom if i == q - 1 else 0

        # This tile *computes* this many output rows before cropping
        H_out_local_patch = conv_out_size_1d(L_local, pad_local_top, pad_local_bot, kh, stride_h)

        # Effective rows counted in MACs = rows actually computed in this tile
        total_rows_effective += H_out_local_patch

    macs_split = C_out * total_rows_effective * W_out * macs_per_out
    return base_macs, macs_split


def estimate_conv_macs_split_HW(
    H, W, C_in, C_out,
    kh, kw,
    stride_h=1, stride_w=1,
    pad_top=0, pad_left=0, pad_bottom=0, pad_right=0,
    q_h=1,
    q_w=1,
    patch_style_overlap=False
):
    """
    Estimate MACs when splitting input into q_h × q_w patches.

    If patch_style_overlap=True, each tile adds halo overlap = kernel - stride,
    so MACs may exceed base MACs due to duplicated boundary computations.
    Otherwise, each output region is computed exactly once ⇒ MACs == base.
    """
    # --- Base output size ---
    H_out = conv_out_size_1d(H, pad_top, pad_bottom, kh, stride_h)
    W_out = conv_out_size_1d(W, pad_left, pad_right, kw, stride_w)

    macs_per_out = C_in * kh * kw
    base_macs = C_out * H_out * W_out * macs_per_out

    # If no split
    if q_h <= 1 and q_w <= 1:
        return base_macs, base_macs

    # Partition output rows and cols across q_h × q_w tiles
    base_rows = H_out // q_h
    rem_rows  = H_out % q_h
    rows_per_tile = [(base_rows + 1 if i < rem_rows else base_rows) for i in range(q_h)]

    base_cols = W_out // q_w
    rem_cols  = W_out % q_w
    cols_per_tile = [(base_cols + 1 if j < rem_cols else base_cols) for j in range(q_w)]

    if not patch_style_overlap:
        # Ideal case: compute each assigned output once
        total_outs = sum(rows_per_tile) * sum(cols_per_tile)
        macs_split = C_out * total_outs * macs_per_out
        return base_macs, macs_split  # == base_macs

    # --- Patching-style overlap accounting ---
    overlap_h = max(0, kh - stride_h)
    overlap_w = max(0, kw - stride_w)

    total_rows_effective = 0
    for i, assigned_rows in enumerate(rows_per_tile):
        for j, assigned_cols in enumerate(cols_per_tile):
            halo_top = overlap_h if i > 0 else 0
            halo_bot = overlap_h if i < q_h - 1 else 0
            halo_left = overlap_w if j > 0 else 0
            halo_right = overlap_w if j < q_w - 1 else 0

            # Local input size required to produce this tile's assigned outputs + halo
            L_local_h = (assigned_rows - 1) * stride_h + kh + halo_top + halo_bot
            L_local_w = (assigned_cols - 1) * stride_w + kw + halo_left + halo_right

            # Local pads: only first/last tiles see real global pads
            pad_local_top    = pad_top if i == 0 else 0
            pad_local_bottom = pad_bottom if i == q_h - 1 else 0
            pad_local_left   = pad_left if j == 0 else 0
            pad_local_right  = pad_right if j == q_w - 1 else 0

            # Outputs actually computed for this tile
            H_out_local = conv_out_size_1d(L_local_h, pad_local_top, pad_local_bottom, kh, stride_h)
            W_out_local = conv_out_size_1d(L_local_w, pad_local_left, pad_local_right, kw, stride_w)

            total_rows_effective += H_out_local * W_out_local

    macs_split = C_out * total_rows_effective * macs_per_out
    return base_macs, macs_split


# def estimate_conv_macs_split(H, W, C_in, C_out, kh, kw, stride, q):
#     """
#     q (int): number of height splits (e.g. 2, 3, 4)

#     Returns:
#         total_macs_split (int): estimated total MACs after splitting
#     """
#     overlap = kh - stride
#     base_h = H // q
#     total_macs_split = 0

#     for i in range(q):
#         # last split may be larger if H not divisible by q
#         h_start = i * base_h
#         h_end = (i + 1) * base_h if i < q - 1 else H
#         local_h = h_end - h_start
        
#         # padding overlap region
#         if i > 0:
#             local_h += overlap
#         if i < q - 1:
#             local_h += overlap

#         # compute output size from this split
#         H_out_local = (local_h - kh) // stride + 1
#         W_out_local = (W - kw) // stride + 1

#         # MACs per split
#         macs_split = C_out * H_out_local * W_out_local * C_in * kh * kw
#         print(f"For {i}/{q} duplicate, macs_split: {macs_split}")
#         total_macs_split += macs_split

#     return total_macs_split


def compute_conv_macs_from_node(node,q):
    if node["op_type"] != "Conv":
        return 0

    # if ENABLE_MICROGRAPH_MODE:
    #     patch_style_overlap = False
    # else:
    #     patch_style_overlap = True
    patch_style_overlap = True

    input_shapes = node["input_shapes"]
    output_shapes = node["output_shapes"]

    N, C_in, H_in, W_in = input_shapes[0]
    N, C_out, H_out, W_out = output_shapes[0]

    # Extract input and output shape
    if input_shapes == [] or output_shapes == []:
        return 0  # Cannot compute without shape info

    attrs = node["attributes"]

    kh, kw   = attrs.get("kernel_shape", [1, 1])
    sh, sw   = attrs.get("strides", [1, 1])
    pt, pl, pb, pr = attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]

    if ENABLE_TINYNAS_MODE:  # 2D TS
        base_macs, split_macs = estimate_conv_macs_split_HW(
            H_in, W_in, C_in, C_out,
            kh, kw,
            stride_h=sh, stride_w=sw,
            pad_top=pt, pad_left=pl, pad_bottom=pb, pad_right=pr,
            q_h=q, q_w=q,
            patch_style_overlap=patch_style_overlap,
        )
    else:     # 1D TS
        base_macs, split_macs = estimate_conv_macs_split_H(
            H_in, W_in, C_in, C_out,
            kh, kw,
            stride_h=sh, stride_w=sw,
            pad_top=pt, pad_left=pl, pad_bottom=pb, pad_right=pr,
            q=q,
            patch_style_overlap=patch_style_overlap,
        )
    # if C_out ==None:
    #     C_out =1
    # if H_out ==None:
    #     H_out =1
    # if W_out ==None:
    #     W_out =1

    #if C_in ==None:
    #    C_in =1
    
    # kh, kw = attrs.get("kernel_shape", [1, 1])
    # stride = attrs.get("strides", [1, 1])[0]
    # pads = attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]
    # pad_top, pad_bottom = pads[0], pads[2]

    #if H_in ==None:
    #    H_in =1
    #if pad_top ==None:
    #    pad_top =0
    #if pad_bottom ==None:
    #    pad_bottom =0 
    
    # H_padded = H_in + pad_top + pad_bottom

    # # Actual number of windows (rows) per split if we split along height
    # num_splits = max(1, (H_padded - kh) // stride + 1)

    # macs_per_window = C_in * kh * kw
    # total_windows = H_out * W_out  # original output
    # base_macs = C_out * total_windows * macs_per_window

    #kernel_shape = attrs.get("kernel_shape", [1, 1])
    #K_h, K_w = kernel_shape

    # if K_h ==None:
    #     K_h =1
    # if K_w ==None:
    #     K_w =1

    #groups = 1

    # if q>1:
    #     total_macs_split = estimate_conv_macs_split(H_in, W_in, C_in, C_out, kh, kw, stride, q)
    # else:
    #     total_macs_split = 0 
    #base_macs = C_out * H_out * W_out * (C_in * K_h * K_w) // groups
    
    return base_macs, split_macs

def compute_conv_access_from_node(node, q, dtype_bytes=1, split_mode="spatial", dim="1d"):
    if node["op_type"] != "Conv":
        return 0

    input_shapes  = node.get("input_shapes") or []
    output_shapes = node.get("output_shapes") or []
    attrs         = node.get("attributes", {})

    # Extract input and output shape
    if not input_shapes or not output_shapes or input_shapes[0] is None or output_shapes[0] is None:
        return 0, 0

    # N, C, H, W (only C_in/C_out matter for param size)
    _, C_in, _, _  = input_shapes[0]
    _, C_out, _, _ = output_shapes[0]

    # if C_out ==None:
    #     C_out =1
    # if H_out ==None:
    #     H_out =1
    # if W_out ==None:
    #     W_out =1

    # if C_in ==None:
    #     C_in =1
    
    kh, kw = attrs.get("kernel_shape", [1, 1])
    groups = attrs.get("group", 1)
    #kernel_shape = attrs.get("kernel_shape", [1, 1])
    #K_h, K_w = kernel_shape

    # if K_h ==None:
    #     K_h =1
    # if K_w ==None:
    #     K_w =1
    # Logical weight size (elements): [C_out, C_in/groups, kh, kw]
    cin_per_group = C_in // max(1, groups)
    weight_elems  = int(C_out) * int(cin_per_group) * int(kh) * int(kw)

    if dim == "1d":
        q_num = max(1, int(q))
    elif dim == "2d":
        q_num = max(1, int(q)) * max(1, int(q))

    split_mode = str(split_mode).lower()

    if split_mode == "spatial":
        # Your scenario: full filters needed for every spatial piece.
        loads = q_num
    elif split_mode in ("cin", "cout"):
        # Channel-partitioned kernels: each tile needs only a fraction of the weights,
        # but across all tiles you read the full tensor exactly once (assuming perfect partitioning).
        loads = 1
    else:
        raise ValueError("split_mode must be 'spatial', 'cin', or 'cout'")

    param_elems = weight_elems * loads

    return param_elems 

def compute_conv_macs_with_split(node, num_sp):
    if node["op_type"] != "Conv":
        return 0

    input_shapes = node["input_shapes"]
    output_shapes = node["output_shapes"]
    attrs = node["attributes"]

    if input_shapes[0] is None or output_shapes[0] is None:
        return 0

    _, C_in, H_in, W_in = input_shapes[0]
    _, C_out, H_out, W_out = output_shapes[0]

    # if C_out ==None:
    #     C_out =1
    # if H_out ==None:
    #     H_out =1
    # if W_out ==None:
    #     W_out =1

    # if C_in ==None:
    #     C_in =1

    #num_splits = H_out
    #H_out =1
    # Extract kernel and stride
    kh, kw = attrs.get("kernel_shape", [1, 1])
    stride = attrs.get("strides", [1, 1])[0]
    pads = attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]
    pad_top, pad_bottom = pads[0], pads[2]

    # if H_in ==None:
    #     H_in =1
    # if pad_top ==None:
    #     pad_top =0
    # if pad_bottom ==None:
    #     pad_bottom =0 
    H_padded = H_in + pad_top + pad_bottom

    # Actual number of windows (rows) per split if we split along height
    num_splits = max(1, (H_padded - kh) // stride + 1)

    macs_per_window = C_in * kh * kw
    total_windows = H_out * W_out  # original output
    base_macs = C_out * total_windows * macs_per_window

    # If duplication is done per split (i.e., same conv logic per slice)
    total_macs = base_macs * num_sp

    return total_macs


def get_subgraph_branch_info(subgraph_nodes, total_sets_of_branches, node_info):
    """
    Given a list of operators in a subgraph, this function determines
    their branch info including set index and branch index.
    Returns a dictionary: {set_idx: {br_idx: [op, ...], ...}, ...}
    """
    branch_info_by_set = {}
    branch_io ={}
    for op in subgraph_nodes:
        for branch_set in total_sets_of_branches:
            op_list = branch_set['operations']
            br_num = branch_set['br_num']
            set_idx = branch_set['br_set_idx']
            branch_io[set_idx]={"input": None, "output": None}

            # Share input or merge node are not part of a numbered branch
            if op == branch_set['shared_inputs_from']:
                branch_io[set_idx]["input"] = op
                continue
            if op == branch_set['merged_by']:
                branch_io[set_idx]["output"] = op
                continue

            if op in op_list:
                lower = max((b for b in br_num if b <= op), default=None)
                upper = min((b for b in br_num if b > op), default=None)
                if lower == br_num[-1]:
                    upper = op_list[-1]
                elif upper and upper > op:
                    upper = upper - 1

                br_idx = br_num.index(lower) + 1

                # Initialize nested dicts if needed
                if set_idx not in branch_info_by_set:
                    branch_info_by_set[set_idx] = {}
                if br_idx not in branch_info_by_set[set_idx]:
                    branch_info_by_set[set_idx][br_idx] = []

                # Add this operator to the correct branch
                branch_info_by_set[set_idx][br_idx].append(op)
                break  # Stop once matched

    return branch_info_by_set, branch_io


def split_by_branch_sets(subgraph_nodes, branch_info):
    all_sets = sorted(branch_info.keys())
    set_ranges = {}

    for set_id in all_sets:
        ops_in_set = []
        for br_ops in branch_info[set_id].values():
            ops_in_set.extend(br_ops)
        set_ranges[set_id] = (min(ops_in_set), max(ops_in_set))

    subgraph_nodes = sorted(subgraph_nodes)
    before_all = []
    between_sets = {}
    after_all = []

    # Handle before the first set
    first_start = set_ranges[all_sets[0]][0]
    before_all = [op for op in subgraph_nodes if op < first_start]

    # Handle between sets
    for i in range(len(all_sets) - 1):
        end_current = set_ranges[all_sets[i]][1]
        start_next = set_ranges[all_sets[i + 1]][0]
        between_sets[(all_sets[i], all_sets[i + 1])] = [
            op for op in subgraph_nodes if end_current < op < start_next
        ]

    # Handle after the last set
    last_end = set_ranges[all_sets[-1]][1]
    after_all = [op for op in subgraph_nodes if op > last_end]

    return before_all, between_sets, after_all

from itertools import permutations, product

def generate_all_subgraph_orders(before_all, between_sets, after_all, branch_info):
    # Step 1: for each set, get fixed branch sequences
    #print(f"branch_info: {branch_info}")
    all_br_orders_per_set = {}
    for set_id, branches in branch_info.items():
        branch_seq = list(branches.values())  # each value is a list of ops in one branch
        all_br_orders_per_set[set_id] = list(permutations(branch_seq))  # permute branch order only
    #print(f"branch_seq: {branch_seq}")
    #print(f"all_br_orders_per_set: {all_br_orders_per_set}")
    # Step 2: combine sets in order
    sorted_set_ids = sorted(all_br_orders_per_set.keys())
    all_set_combinations = list(product(*(all_br_orders_per_set[set_id] for set_id in sorted_set_ids)))

    # Step 3: build full op sequence with before/between/after
    final_orders = []
    for combo in all_set_combinations:
        current_order = before_all.copy()

        for i, set_id in enumerate(sorted_set_ids):
            for branch in combo[i]:
                current_order.extend(branch)  # keep ops in branch in original order

            # Insert between-set ops
            if i < len(sorted_set_ids) - 1:
                key = (sorted_set_ids[i], sorted_set_ids[i + 1])
                if key in between_sets:
                    current_order.extend(between_sets[key])

        current_order.extend(after_all)
        final_orders.append(current_order)
        
        #print(f"current_order: {current_order}")
    #print(f"final_orders: {final_orders}")

    return final_orders

def calculate_subgraph_memory_usage(
    subgraph_op_order,
    node_info,
    data_usage,
    total_sets_of_branches
):
    """
    Given an operator order in a subgraph and precomputed node_info and data_usage,
    compute IFM, OFM, and buffer sizes considering branch structure.

    Assumes subgraph_op_order is correctly constructed (e.g., via permutations over branches).
    """
    op_to_idx = {op: i for i, op in enumerate(subgraph_op_order)}  # local index in subgraph
    num = len(subgraph_op_order)
    ifm_sizes = [0] * num
    ofm_sizes = [0] * num
    buffer_sizes = [0] * num

    # === Compute IFM and OFM sizes from data_usage ===
    for i, op in enumerate(subgraph_op_order):
        node = node_info[op]



        ifm_sizes[i] = sum(data_usage[di]["size"] for di in node.get("data_in", []))
        ofm_sizes[i] = sum(data_usage[di]["size"] for di in node.get("data_out", []))

    # === Compute Buffer Sizes based on branch structure ===
    ofm_finished = 0
    ifm_retain = 0
    #print(f"subgraph_op_order")

    for i, op in enumerate(subgraph_op_order):
        if node['op_type'] in unsplittable_ops:
            continue

        #print(f"op: {op}")
        br, op_list, br_cnt = find_op_in_br(op, total_sets_of_branches)

        if br is None:
            buffer_sizes[i] = 0
            ofm_finished = 0
        else:
            buffer_sizes[i] = 0
            if br == 1:
                ofm_finished = 0
                if op == op_list[0]:
                    ifm_retain = ifm_sizes[i]
                else:
                    buffer_sizes[i] = ifm_retain + ofm_finished

                if op == op_list[-1]:
                    ofm_finished = ofm_sizes[i]
            
            elif br == br_cnt:   #last br in a set
                    buffer_sizes[i] = ofm_finished

            else:
                if op == op_list[0]:
                    ifm_retain = ifm_sizes[i]
                    buffer_sizes[i] = ofm_finished
                else:
                    buffer_sizes[i] = ifm_retain + ofm_finished

                if op == op_list[-1]:
                    ofm_finished += ofm_sizes[i]

    return {
        'ifm': ifm_sizes,
        'ofm': ofm_sizes,
        'buf': buffer_sizes,
        'total': [ifm + ofm + buf for ifm, ofm, buf in zip(ifm_sizes, ofm_sizes, buffer_sizes)],
        'peak': max(ifm_sizes[i] + ofm_sizes[i] + buffer_sizes[i] for i in range(num)),
        'peak_idx': subgraph_op_order[
            max(range(num), key=lambda i: ifm_sizes[i] + ofm_sizes[i] + buffer_sizes[i])
        ]
    }

def find_related_shared_merge_ops(best_d, total_sets_of_branches, branch_info):
    shared_inputs = []
    merged_outputs = []

    for br_set in total_sets_of_branches:
        shared_op = br_set["shared_inputs_from"]
        merge_op = br_set["merged_by"]
        if shared_op in best_d:
            shared_inputs.append(shared_op)
        if merge_op in best_d:
            merged_outputs.append(merge_op)

    op_to_branch = {}
    for set_id in branch_info:
        for br_ops in branch_info[set_id].values():
            for op in br_ops:
                op_to_branch[op] = br_ops

    seen = set()
    ordered_branches = []
    for op in best_d:
        if op in op_to_branch:
            br = tuple(op_to_branch[op])  # use tuple to hash
            if br not in seen:
                ordered_branches.append(list(br))
                seen.add(br)

    return shared_inputs, merged_outputs, ordered_branches


def search_dxq(subgraph_nodes, data_usage, node_info, mem_usage, M_target, total_sets_of_branches):
    valid_q = VALID_Q 
    regular_d_candidates = []
    d_candidates = []
    q_candidates =[]

    chosed_config = {'d': None, 'q': None, 'peak_mem': None, 'total_mem_under_q': None,
                     'q_candidate': None, 'd_candidate_num': None,'under_mem': False}

    for q in valid_q:
        valid = True
        for op in subgraph_nodes:
            if node_info[op]['op_type'] in unsplittable_ops:
                continue

            if(node_info[op]['input_shapes'][0][2] < q or node_info[op]['output_shapes'][0][2] < q):
                valid = False
            #print(node_info[op]['input_shapes'][0])
            elif (node_info[op]['input_shapes'][0][2] %q != 0) or (node_info[op]['input_shapes'][0][2] /q <= 1):
                valid = False
            #print(node_info[op]['output_shapes'][0])
            elif (node_info[op]['output_shapes'][0][2] %q != 0) or (node_info[op]['output_shapes'][0][2] /q <= 1):
                valid = False

        if valid:
            q_candidates.append(q)

    #q_candidates = Q_LIST
    #print("q_candidates: ", q_candidates)

    #test
    #subgraph_nodes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #subgraph_nodes=list(range(1,31))

    #print("find dxq")
    branch_info= []
    branch_io = []
    #print(f"subgraph_nodes:{subgraph_nodes}")
    branch_info, branch_io = get_subgraph_branch_info(subgraph_nodes, total_sets_of_branches, node_info)

    before_all, between_sets, after_all = split_by_branch_sets(subgraph_nodes, branch_info)
    orders = generate_all_subgraph_orders(before_all, between_sets, after_all, branch_info)
    d_candidates = orders
    
    #print("branch_info:", branch_info)
    #print("Total orders:", len(d_candidates))
    #print("q_candidates:", q_candidates)
    
    best_d = None
    best_peak = float('inf')
    best_sub_mem = None

    #sub_mem_usage = calculate_subgraph_memory_usage(d_candidates[0], node_info, data_usage, total_sets_of_branches)
    
    #print(f"try {len(d_candidates)} d candidates")
    for d in d_candidates:

        sub_mem_usage = calculate_subgraph_memory_usage(d, node_info, data_usage, total_sets_of_branches)
        #print("sub_mem_usage['peak']: ", sub_mem_usage['peak'])
        M=sub_mem_usage['peak']
        if M < best_peak:
            best_d = d
            best_peak = M
            best_sub_mem = sub_mem_usage

    #print ('best_d: ', best_d)
    #print ('best_peak: ', best_peak)
    #print ('best_sub_mem: ', best_sub_mem)
    peak_op = best_sub_mem['peak_idx']
    
    #print(peak_op)
    shared_ops, merge_ops, ordered_branches = find_related_shared_merge_ops(best_d, total_sets_of_branches, branch_info)
    #print("Shared Inputs:", shared_ops)
    #print("Merged Outputs:", merge_ops)
    #print("Ordered_branches:", ordered_branches)
    
    share_input_size = data_usage[node_info[shared_ops[0]]["data_in"][0]]["size"]
    br_output_size=[]
    for br in ordered_branches:
        per_br_output_size = data_usage[node_info[br[-1]]["data_out"][0]]["size"]
        br_output_size.append(per_br_output_size)


    #print("share_input_size: ", share_input_size)
    #print("br_output_size: ", br_output_size)
    ## 0506 go from this
    #print("q_candidates: ", q_candidates)
    #print(f"try {len(q_candidates)} q candidates")
    for q in q_candidates:
        total_branches = len(br_output_size)
        base_sum = sum(br_output_size)
        ret_out_total = []
        ret_in = [0]*q
        total_mem_under_q = []
        #print("Iter: q=", q)

        for a in range(0, q):
            ret_out = [0] * total_branches
            ret_in[a] = share_input_size // q * (q - a-1)
            for br in range(total_branches):
                prefix_sum = sum(br_output_size[:br])  # sum of outputs from previous branches
                suffix_sum = base_sum if a > 0 else 0  # base sum used only from a >= 2
                ret_out[br] = suffix_sum // q *(a) + prefix_sum // q

            ret_out_total.append(ret_out)

            shared_op_ret_a = share_input_size // q * (q - a-1) + suffix_sum // q *(a)
            shared_op_io_a = share_input_size // q + (data_usage[node_info[shared_ops[0]]["data_out"][0]]["size"])// q
            total_mem_under_q.append(shared_op_ret_a+shared_op_io_a)
            
            for bidx, br in enumerate(ordered_branches):
                for op in br:
                    ifm = data_usage[node_info[op]["data_in"][0]]["size"]
                    ofm = data_usage[node_info[op]["data_out"][0]]["size"]
                    total = ifm//q + ofm//q + ret_out[bidx]
                    total_mem_under_q.append(total)
            sum_out = base_sum // q *(a+1) + share_input_size // q * (q - a-1)
            total_mem_under_q.append(sum_out)

        #print("ret_in:", ret_in)
        #print("ret_out_total:", ret_out_total)
        #print("total_mem_under_q:", total_mem_under_q)

        peak_under_q = max(total_mem_under_q)
        #print("max:", peak_under_q)

        if GOAL_PRIORITY == 'mem':

            if chosed_config['peak_mem'] == None:
                chosed_config['d'] = best_d
                chosed_config['q'] = q
                chosed_config['peak_mem'] = peak_under_q
                chosed_config['total_mem_under_q'] = total_mem_under_q
                chosed_config['q_candidate'] = q_candidates
                chosed_config['d_candidate_num'] = d_candidates
                chosed_config['under_mem'] = False
            else:
                if peak_under_q < chosed_config['peak_mem']:
                    chosed_config['d'] = best_d
                    chosed_config['q'] = q
                    chosed_config['peak_mem'] = peak_under_q
                    chosed_config['total_mem_under_q'] = total_mem_under_q
                    chosed_config['q_candidate'] = q_candidates
                    chosed_config['d_candidate_num'] = d_candidates
                    chosed_config['under_mem'] = False
                else:
                    continue
        else:
            if peak_under_q <= M_target:
                #print(f"Find dxq under M_target: d: {d}  q: {q} \n")
                return {
                    'd': best_d,
                    'q': q,
                    'peak_mem': peak_under_q,
                    'total_mem_under_q': total_mem_under_q,
                    'q_candidate': q_candidates,
                    'd_candidate_num': d_candidates,
                    'under_mem': True

                }

    #print(f"Did not find dxq under M_target, the last tried d: {d}  q: {q} \n")
    

    if GOAL_PRIORITY == 'mem':
        if chosed_config['peak_mem'] != None:
            if int(chosed_config['peak_mem']) > M_target:
                #print("not fit vm: ", chosed_config['peak_mem'])
                return None
            else:
                chosed_config['under_mem'] = True
                #print("check: return config")
                return chosed_config
    
        else:
            return None

    # {    
    #     'd': best_d,
    #     'q': q,
    #     'peak_mem': peak_under_q,
    #     'total_mem_under_q': total_mem_under_q,
    #     'q_candidate': q_candidates,
    #     'd_candidate_num': d_candidates,
    #     'under_mem': False
    #     }

def search_dxq_single_branch(subgraph_nodes, data_usage, node_info, mem_usage, M_target):

    valid_q = VALID_Q 
    q_candidates =[]
    chosed_config = {'d': None, 'q': None, 'peak_mem': None, 'total_mem_under_q': None,
                     'q_candidate': None, 'd_candidate_num': 1,'under_mem': False}



    for q in valid_q:
        valid = True
        for op in subgraph_nodes:
            if(node_info[op]['input_shapes'][0][2] < q or node_info[op]['output_shapes'][0][2] < q):
                valid = False
            #print(node_info[op]['input_shapes'][0])
            elif (node_info[op]['input_shapes'][0][2] %q != 0) or (node_info[op]['input_shapes'][0][2] /q <= 1):
                valid = False
            #print(node_info[op]['output_shapes'][0])
            elif (node_info[op]['output_shapes'][0][2] %q != 0) or (node_info[op]['output_shapes'][0][2] /q <= 1):
                valid = False
        if valid:
            q_candidates.append(q)
    #q_candidates = Q_LIST
    #print("q_candidates: ", q_candidates)


    # Assume fixed order (no permutation)
    best_d = subgraph_nodes
    best_peak = float('inf')
    ifm_sizes = []
    ofm_sizes = []
    buf_sizes = []
    sub_mem_usage = {}

    for op in best_d:
        ifm_sizes.append(mem_usage['ifm'][op])
        ofm_sizes.append(mem_usage['ofm'][op])
        buf_sizes.append(mem_usage['buf'][op])
        
    sub_mem_usage['ifm'] = ifm_sizes
    sub_mem_usage['ofm'] = ofm_sizes
    sub_mem_usage['buf'] = buf_sizes
    sub_mem_usage['total'] = [ifm + ofm + buf for ifm, ofm, buf in zip(ifm_sizes, ofm_sizes, buf_sizes)]
    sub_mem_usage['peak'] = max(ifm_sizes[i] + ofm_sizes[i] + buf_sizes[i] for i in range(len(best_d)))
    sub_mem_usage['peak_idx'] = best_d[
            max(range(len(best_d)), key=lambda i: ifm_sizes[i] + ofm_sizes[i] + buf_sizes[i])
        ]

    #best_sub_mem = calculate_subgraph_memory_usage(best_d, node_info, data_usage)
    best_peak = sub_mem_usage['peak']
    peak_op = sub_mem_usage['peak_idx']

    #print("Single-branch best_d:", best_d)
    #print("Base peak:", best_peak)

    # Get shared input and output sizes
    start = best_d[0]
    end = best_d[-1]
    
    start_size = data_usage[node_info[start]["data_in"][0]]["size"]
    end_size = data_usage[node_info[end]["data_out"][0]]["size"]
    #print("start_size", start_size)
    #print("end_size", end_size)
    

    for q in q_candidates:
        total_mem_under_q = []

        for a in range(q):
            # Shared input retention + output retention
            start_ret = start_size // q * (q - a - 1)
            end_ret = end_size // q * a
            #print("start_ret, end_ret : ", start_ret, end_ret)

            # Internal op memory usage
            for op in subgraph_nodes:
                ifm = data_usage[node_info[op]["data_in"][0]]["size"]
                ofm = data_usage[node_info[op]["data_out"][0]]["size"]
                total = ifm // q + ofm // q + start_ret + end_ret
                total_mem_under_q.append(total)

        peak_under_q = max(total_mem_under_q)
        #print(f"[q={q}] peak_mem:", peak_under_q)

        if GOAL_PRIORITY == 'mem':

            if peak_under_q <= M_target:
                if not chosed_config['under_mem']:
                    chosed_config['d'] = best_d
                    chosed_config['q'] = q
                    chosed_config['peak_mem'] = peak_under_q
                    chosed_config['total_mem_under_q'] = total_mem_under_q
                    chosed_config['q_candidate'] = q_candidates
                    chosed_config['d_candidate_num'] = 1
                    chosed_config['under_mem'] = True
                else:
                    if peak_under_q < chosed_config['peak_mem']:
                        chosed_config['d'] = best_d
                        chosed_config['q'] = q
                        chosed_config['peak_mem'] = peak_under_q
                        chosed_config['total_mem_under_q'] = total_mem_under_q
                        chosed_config['q_candidate'] = q_candidates
                        chosed_config['d_candidate_num'] = 1
                        chosed_config['under_mem'] = True
                    else:
                        continue
        else:
            if peak_under_q <= M_target:
                #print(f"Find dxq under M_target: d: {d}  q: {q} \n")
                return {
                    'd': best_d,
                    'q': q,
                    'peak_mem': peak_under_q,
                    'total_mem_under_q': total_mem_under_q,
                    'q_candidate': q_candidates,
                    'd_candidate_num': 1,
                    'under_mem': True

                }

    #print(f"Did not find dxq under M_target, the last tried d: {d}  q: {q} \n")
    

    if GOAL_PRIORITY == 'mem':
        if not chosed_config['under_mem']:
            return None
        else:
            return chosed_config
    
    else:
        return None
    # {    
    #     'd': best_d,
    #     'q': q,
    #     'peak_mem': peak_under_q,
    #     'total_mem_under_q': total_mem_under_q,
    #     'q_candidate': q_candidates,
    #     'd_candidate_num': 1,
    #     'under_mem': False
    #     }






def select_configurations(node_list, op_order, data_usage, node_info, mem_usage, M_target, total_sets_of_branches, GOAL_PRIORITY):
    global unsplittable_ops
    dup_path = []
    est_peak_per_path = []
    q_list_per_path = []
    total_pdq_config = []

    cannot_dup = []
    set_of_op_start_br = []
    set_of_op_end_br = []

    # Helper to key a branch-set once
    def _br_key(br):
        return (br['shared_inputs_from'], br['merged_by'])



    for setbr in total_sets_of_branches:
        set_of_op_start_br.append(setbr['shared_inputs_from'])
        set_of_op_end_br.append(setbr['merged_by'])
    
    #remaining_peaks = sorted(peak_list, key=lambda x: mem_usage['total'][x], reverse=True)
    #print(remaining_peaks)
    selected = []

    time_record={"multi":[], "single":[]}
    
    #print("set_of_op_start_br: ", set_of_op_start_br)
    #print("set_of_op_end_br: ", set_of_op_end_br)


    if GOAL_PRIORITY == 'bal':
        remaining_peaks = sorted(op_order, key=lambda x: mem_usage['total'][x], reverse=True)


        while remaining_peaks:

            #print("++ Already used op: ", selected)
            peak_op = remaining_peaks.pop(0)
            #print("current peak op: ", peak_op)
            if peak_op in selected:
                continue

            init_subgraph = [peak_op]
            branch_set=None
            if peak_op in set_of_op_start_br:
                for major in total_sets_of_branches:
                    if major['shared_inputs_from'] == peak_op:
                        br_idx = major['br_set_idx']
                        break
                    if peak_op in major['operations']:
                        br_idx = major['br_set_idx']
                        break
                #print("peak_op:", peak_op, "br_idx: ", br_idx)
                br_set = total_sets_of_branches[br_idx]



                #print("set_of_op_start_br: ", set_of_op_start_br)
                #br_set = total_sets_of_branches[set_of_op_start_br.index(peak_op)]
                #print("br_set: ", br_set)
                branch_set = br_set

            if peak_op in set_of_op_end_br:
                for major in total_sets_of_branches:
                    if major['merged_by'] == peak_op:
                        br_idx = major['br_set_idx']
                        break
                    if peak_op in major['operations']:
                        br_idx = major['br_set_idx']
                        break
                #print("peak_op:", peak_op, "br_idx: ", br_idx)
                br_set = total_sets_of_branches[br_idx]



                #print("set_of_op_start_br: ", set_of_op_start_br)
                #br_set = total_sets_of_branches[set_of_op_start_br.index(peak_op)]
                #print("br_set: ", br_set)
                branch_set = br_set


            best_subgraph = init_subgraph
            #plus_op =[]
            best_config = None
            unwell_config = None
            config = None
            best_peak = mem_usage['total'][peak_op]
            total_ops = float('inf')

            while True:
                left=[]
                right=[]
                data_in = node_info[best_subgraph[0]]['data_in']
                data_out = node_info[best_subgraph[-1]]['data_out']
                
                for nidx, node in enumerate(node_info):
                    for din in data_in:
                        if din in node['data_out']:
                            left.append(nidx)
                    for dout in data_out:
                        if dout in node['data_in']:
                            right.append(nidx)

                #print("peak_op:", peak_op)
                #print("left:", left, "  right:", right)

                if right and right[0] in set_of_op_start_br:
                    right = []
                if right and right[0] in unsplittable_ops:
                    right = []

                if left ==[] and right ==[]:
                    break

                branch_set = []

                if len(left) > 1:  
                    for major in total_sets_of_branches:
                        if major['merged_by'] == best_subgraph[0]:
                            br_idx = major['br_set_idx']
                            break
                        if best_subgraph[0] in major['operations']:
                            br_idx = major['br_set_idx']
                            break
                        #print("op:", op, "br_idx: ", br_idx)
                    br_set = total_sets_of_branches[br_idx]
                    branch_set = br_set
                    left = []
                    right = []

                elif len(left) ==1 and left[0] > 0:
                    pre_node = data_usage[node_info[left[0]]['data_in'][0]]['from'][0]
                    if pre_node in set_of_op_start_br and node_info[pre_node]['op_type'] in unsplittable_ops:
                        for major in total_sets_of_branches:
                            if major['shared_inputs_from'] == pre_node:
                                br_idx = major['br_set_idx']
                                break
                            if left[0] in major['operations']:
                                br_idx = major['br_set_idx']
                                break
                                #print("op:", op, "br_idx: ", br_idx)
                        br_set = total_sets_of_branches[br_idx]
                        branch_set = br_set
                        left = []
                        right = []

                for op in left:
                    if op in set_of_op_start_br:
                        for major in total_sets_of_branches:
                            if major['shared_inputs_from'] == op:
                                br_idx = major['br_set_idx']
                                break
                            if op in major['operations']:
                                br_idx = major['br_set_idx']
                                break
                        #print("op:", op, "br_idx: ", br_idx)
                        br_set = total_sets_of_branches[br_idx]
                        
                        branch_set = br_set

                for op in right:
                    if op in set_of_op_end_br:
                        for major in total_sets_of_branches:
                            if major['merged_by'] == op:
                                br_idx = major['br_set_idx']
                                break
                            if op in major['operations']:
                                br_idx = major['br_set_idx']
                                break
                        #print("op:", op, "br_idx: ", br_idx)
                        br_set = total_sets_of_branches[br_idx]
                        
                        branch_set = br_set

                

                if branch_set:
                    for nop in branch_set['operations']:
                        if nop in selected:
                            break


                    new_subgraph = list(range(branch_set['shared_inputs_from'], branch_set['merged_by']+1))
                    for sop in best_subgraph:
                        if sop not in new_subgraph:
                            new_subgraph.append(sop)

                    searching_time_multi =0
                    start_time = datetime.now()
                    config = search_dxq(new_subgraph, data_usage, node_info, mem_usage, M_target, total_sets_of_branches)
                    end_time = datetime.now()
                    searching_time_multi=end_time-start_time
                    time_record["multi"].append(searching_time_multi)
                    

                else:  # branch_set=None
                    for lop in left:
                        if lop in selected:
                            left = []
                            break
                    for rop in right:
                        if rop in selected:
                            right = []
                            break
                    if left == [] and right == []:
                        config = None
                        break

                    new_subgraph = left + best_subgraph + right
                    searching_time_single =0
                    start_time_single = datetime.now()
                    config = search_dxq_single_branch(new_subgraph, data_usage, node_info, mem_usage, M_target)
                    end_time_single = datetime.now()
                    searching_time_single = end_time_single-start_time_single
                    time_record["single"].append(searching_time_single)
                    
                
                if config:
                    #if config['under_mem']:
                    ops = config['q'] * len(config['d'] )
                    M_peak = config['peak_mem']


                    if (best_peak > M_target and best_peak >= M_peak) or (best_peak <= M_target and M_peak <= M_target and ops <= total_ops):
                        best_config = config
                        best_peak = M_peak
                        total_ops = ops
                        best_subgraph = new_subgraph
                        branch_set=None

                    else:
                        break
                    
                    #     unwell_config = config
                    #     last_graph = new_subgraph
                    
                else:
                    break

                

            if best_config:
                total_pdq_config.append({
                    'subgraph': best_subgraph,
                    'peak': peak_op,
                    'ori_peak_mem': mem_usage['total'][peak_op],
                    'after_peak_mem': best_config['peak_mem'],
                    'dup_config': best_config,
                })
                #est_peak_per_path.append(best_peak)
                #q_list_per_path.append(best_config['q'])
                for n in best_subgraph:
                    selected.append(n)

                    if n in remaining_peaks:
                        remaining_peaks.remove(n)
                    if n in cannot_dup:
                        cannot_dup.remove(n)


            # elif unwell_config:
            # #print("remaining_peaks: ", remaining_peaks)
            #     total_pdq_config.append({
            #         'subgraph': new_subgraph,
            #         'peak': peak_op,
            #         'ori_peak_mem': mem_usage['total'][peak_op],
            #         'after_peak_mem': unwell_config['peak_mem'],
            #         'dup_config': unwell_config,
            #     })
            #     cannot_dup.append(peak_op)

            else:
                total_pdq_config.append({
                    'subgraph': None,
                    'peak': peak_op,
                    'ori_peak_mem': mem_usage['total'][peak_op],
                    'after_peak_mem': mem_usage['total'][peak_op],
                    'dup_config': None,
                })
                cannot_dup.append(peak_op)


    if GOAL_PRIORITY == 'mem':
        op_order_by_peak = op_order
        
        visited_subgraphs = set()
        tried_branch_ranges = set()          # {(shared_inputs_from, merged_by), ...}
        #print(f"total_sets_of_branches: {total_sets_of_branches}")

        while True:
            
            #print("op_order_by_peak: ", op_order_by_peak)
            #print("++ Already used op: ", selected)
            peak_op = op_order_by_peak.pop(0)
            #print("current peak op: ", peak_op)
            if peak_op in selected:
                continue

            init_subgraph = [peak_op]
            branch_set=None
            if peak_op in set_of_op_start_br:
                for major in total_sets_of_branches:
                    if major['shared_inputs_from'] == peak_op:
                        br_idx = major['br_set_idx']
                        break
                    if peak_op in major['operations']:
                        br_idx = major['br_set_idx']
                        break
                #print("peak_op:", peak_op, "br_idx: ", br_idx)
                br_set = total_sets_of_branches[br_idx]



                #print("set_of_op_start_br: ", set_of_op_start_br)
                #br_set = total_sets_of_branches[set_of_op_start_br.index(peak_op)]
                #print("br_set: ", br_set)
                branch_set = br_set

            if peak_op in set_of_op_end_br:
                for major in total_sets_of_branches:
                    if major['merged_by'] == peak_op:
                        br_idx = major['br_set_idx']
                        break
                    if peak_op in major['operations']:
                        br_idx = major['br_set_idx']
                        break
                #print("peak_op:", peak_op, "br_idx: ", br_idx)
                br_set = total_sets_of_branches[br_idx]



                #print("set_of_op_start_br: ", set_of_op_start_br)
                #br_set = total_sets_of_branches[set_of_op_start_br.index(peak_op)]
                #print("br_set: ", br_set)
                branch_set = br_set


            best_subgraph = init_subgraph
            #plus_op =[]
            best_config = None
            unwell_config = None
            config = None
            best_peak = mem_usage['total'][peak_op]
            total_ops = float('inf')

            carry_set = 0

            while True:
                left=[]
                right=[]
                data_in = node_info[best_subgraph[0]]['data_in']
                data_out = node_info[best_subgraph[-1]]['data_out']
                
                for nidx, node in enumerate(node_info):
                    for din in data_in:
                        if din in node['data_out']:
                            left.append(nidx)
                    for dout in data_out:
                        if dout in node['data_in']:
                            right.append(nidx)

                #print("peak_op:", peak_op)
                

                if right and right[0] in set_of_op_start_br:
                    right = []
                if right and right[0] in unsplittable_ops:
                    right = []

                if left ==[] and right ==[]:
                    break

                branch_set = []

                if len(left) > 1:  
                    for major in total_sets_of_branches:
                        if major['merged_by'] == best_subgraph[0]:
                            br_idx = major['br_set_idx']
                            break
                        if best_subgraph[0] in major['operations']:
                            br_idx = major['br_set_idx']
                            break
                        #print("op:", op, "br_idx: ", br_idx)
                    br_set = total_sets_of_branches[br_idx]
                    branch_set = br_set
                    left = []
                    right = []

                elif len(left) ==1 and left[0] > 0:
                        pre_node = data_usage[node_info[left[0]]['data_in'][0]]['from'][0]
                        if pre_node in set_of_op_start_br and node_info[pre_node]['op_type'] in unsplittable_ops:
                            for major in total_sets_of_branches:
                                if major['shared_inputs_from'] == pre_node:
                                    br_idx = major['br_set_idx']
                                    break
                                if left[0] in major['operations']:
                                    br_idx = major['br_set_idx']
                                    break
                                #print("op:", op, "br_idx: ", br_idx)
                            br_set = total_sets_of_branches[br_idx]
                            branch_set = br_set
                            left = []
                            right = []
                            #print("branch_set: ", branch_set)
                            
                    #print("left:", left, "  right:", right)
                    
                for op in left:
                    if op in set_of_op_start_br:
                        for major in total_sets_of_branches:
                            if major['shared_inputs_from'] == op:
                                br_idx = major['br_set_idx']
                                break
                            if op in major['operations']:
                                br_idx = major['br_set_idx']
                                break
                        #print("op:", op, "br_idx: ", br_idx)
                        br_set = total_sets_of_branches[br_idx]
                        branch_set = br_set

                for op in right:
                    if op in set_of_op_end_br:
                        for major in total_sets_of_branches:
                            if major['merged_by'] == op:
                                br_idx = major['br_set_idx']
                                break
                            if op in major['operations']:
                                br_idx = major['br_set_idx']
                                break
                        #print("op:", op, "br_idx: ", br_idx)
                        br_set = total_sets_of_branches[br_idx]
                        branch_set = br_set

                #print("branch_set: ", branch_set)
    
                if branch_set:

                    for nop in branch_set['operations']:
                        if nop in selected:
                            break
                        if node_info[nop]['op_type'] in unsplittable_ops:
                            break 

                    key = _br_key(branch_set)
                    if key in tried_branch_ranges:
                        branch_set = []
                        break
                    else:
                        tried_branch_ranges.add(key)

                    
                    #print(f"branch_set: {branch_set}")
                    branch_range = range(branch_set['shared_inputs_from'], branch_set['merged_by'] + 1)
                    new_subgraph = list(branch_range)
                    #print(f"new_subgraph: {new_subgraph}")
                    if carry_set < 1:
                        carry_set += 1
                        for sop in best_subgraph:
                            if sop not in new_subgraph:
                                new_subgraph.append(sop)


                    searching_time_multi =0
                    start_time = datetime.now()
                    #print(f"find dxq for new_subgraph: {new_subgraph}")
                    config = search_dxq(new_subgraph, data_usage, node_info, mem_usage, M_target, total_sets_of_branches)
                    end_time = datetime.now()
                    #print(f"config : {config}")
                    searching_time_multi=end_time-start_time
                    time_record["multi"].append(searching_time_multi)
                    

                else:  # branch_set=None
                    for lop in left:
                        if lop in selected:
                            left = []
                            break
                    for rop in right:
                        if rop in selected:
                            right = []
                            break
                    if left == [] and right == []:
                        config = None
                        break

                    new_subgraph = left + best_subgraph + right
                    searching_time_single =0
                    start_time_single = datetime.now()
                    config = search_dxq_single_branch(new_subgraph, data_usage, node_info, mem_usage, M_target)
                    end_time_single = datetime.now()
                    searching_time_single = end_time_single-start_time_single
                    time_record["single"].append(searching_time_single)
                    
                # ---------- PROGRESS GUARD (prevents infinite loop) ----------
                sig = frozenset(new_subgraph)
                if sig in visited_subgraphs:
                    # we've already analyzed exactly this subgraph; stop expanding
                    break
                visited_subgraphs.add(sig)

                if config:
                    ops = config['q'] * len(config['d'] )
                    M_peak = config['peak_mem']

                    
                    #print("new_subgraph find config: ", new_subgraph)

                    if (best_peak > M_peak):
                        best_config = config
                        best_peak = M_peak
                        total_ops = ops
                        best_subgraph = new_subgraph

                        if branch_set and carry_set == 1:
                            carry_set = 0
                            break
                        else:
                            if branch_set:
                                carry_set +=1
                                branch_set = None
                        
                    else:
                        if branch_set and carry_set == 1:
                            carry_set = 0
                            break
                        else:
                            best_subgraph = new_subgraph
                            if branch_set:
                                carry_set +=1
                                branch_set = None

                        
                    unsplit=False
                    for opn in new_subgraph:
                        if node_info[opn]['op_type'] in unsplittable_ops:
                            #print("with unsplittable_ops")
                            unsplit = True
                            break
                    
                    if unsplit:
                        break
                    
                    
                else:
                    #print("new_subgraph: ", new_subgraph)
                    unsplit=False
                    for opn in new_subgraph:
                        if node_info[opn]['op_type'] in unsplittable_ops:
                            #print("with unsplittable_ops")
                            unsplit = True
                            break
                    if unsplit:
                        break
                    
                    if branch_set and carry_set == 2:
                        carry_set = 0
                        break
                    else:
                        best_subgraph = new_subgraph
                        if branch_set:
                            carry_set +=1
                            branch_set = None
                

                

            if best_config:
                #print(f"'subgraph': {best_subgraph},'best_config': {best_config},")
                total_pdq_config.append({
                    'subgraph': best_subgraph,
                    'peak': peak_op,
                    'ori_peak_mem': mem_usage['total'][peak_op],
                    'after_peak_mem': best_config['peak_mem'],
                    'dup_config': best_config,
                })
                #est_peak_per_path.append(best_peak)
                #q_list_per_path.append(best_config['q'])
                for n in best_subgraph:
                    selected.append(n)

                    if n in op_order_by_peak:
                        op_order_by_peak.remove(n)
                    #if n in cannot_dup:
                    #    cannot_dup.remove(n)

                if op_order_by_peak:
                    next_peak = op_order_by_peak[0]

                    do_next_pdq = True
                    for saved in total_pdq_config:

                        if saved['after_peak_mem'] >= mem_usage['total'][next_peak]:
                            do_next_pdq = False
                            cannot_dup.append(next_peak)
                            break

                    if not do_next_pdq:
                        #print("!! Exit PDQ-MEM selection !!  Peak memory cannot be reduced further since the current saved PDQ-TS remains above the next-peak.")
                        break
                else:
                    #print("!! Exit PDQ-MEM selection !!  All operations are used for PDQ-TS.")
                    break


            else:
                if config:
                    #print(f"'subgraph': {best_subgraph},'dup_config': {config},")
                    total_pdq_config.append({
                        'subgraph': best_subgraph,
                        'peak': peak_op,
                        'ori_peak_mem': mem_usage['total'][peak_op],
                        'after_peak_mem': config['peak_mem'],
                        'dup_config': config,
                    })
                    cannot_dup.append(peak_op)
                
                #print("!! Exit PDQ-MEM selection !!  Peak memory cannot be reduced further since the current-peak cannot find a valid PDQ-TS.")
                break




    return total_pdq_config, time_record, cannot_dup






def get_io_info(model):
    input_info, output_info = [], []
    for i in model.graph.input:
        input_info.append((i.name, [dim.dim_value for dim in i.type.tensor_type.shape.dim]))
    for o in model.graph.output:
        output_info.append((o.name, [dim.dim_value for dim in o.type.tensor_type.shape.dim]))
    return input_info, output_info

def get_op_attributes(node):
    attributes = {
        'stride': None,
        'pads': None,
        'kernel_shape': None,
    }

    for attr in node.attribute:
        if attr.name == 'strides':
            attributes['stride'] = attr.ints
        elif attr.name == 'pads':
            attributes['pads'] = attr.ints
        elif attr.name == 'kernel_shape':
            attributes['kernel_shape'] = attr.ints
    return attributes

def get_tensor_shapes(value_info):
    """Helper function to extract tensor shape from ValueInfoProto."""
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.dim_value > 0 else None)
    return shape

def get_node_io_info(inferred_model):
    node_info = []
    tensor_shapes = {}

    # Populate tensor shapes from inferred model graph value info
    for value_info in inferred_model.graph.value_info:
        tensor_shapes[value_info.name] = get_tensor_shapes(value_info)

    # Include model inputs and outputs shapes
    for input_info in inferred_model.graph.input:
        tensor_shapes[input_info.name] = get_tensor_shapes(input_info)
    for output_info in inferred_model.graph.output:
        tensor_shapes[output_info.name] = get_tensor_shapes(output_info)

    for node in inferred_model.graph.node:
        attributes = get_op_attributes(node)

        input_shapes = [tensor_shapes.get(inp, None) for inp in node.input]
        output_shapes = [tensor_shapes.get(out, None) for out in node.output]

        if node.input !=[]:
            node_info.append({
                "node_name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "attributes": attributes,
                "data_in": [],
                "data_out": [],
                "br_label":"None"
            })

    total_nodes = len(node_info)
    return node_info, total_nodes

def calculate_tensor_size(dimensions):
    """Calculate the size of a tensor given its dimensions."""
    if None in dimensions or 0 in dimensions:
        return 0  # Handle unknown dimensions
    return np.prod(dimensions)

def identify_shared_inputs(node_info):
    """Identify nodes that share inputs across branches."""
    shared_inputs = {}
    for idx, node in enumerate(node_info):
        if not node["inputs"]:  # Skip nodes with no inputs
            continue

        input_name = node["inputs"][0]
        if input_name not in shared_inputs:
            shared_inputs[input_name] = []
        shared_inputs[input_name].append(idx)
    # Keep only inputs used by more than one node (indicating shared input)
    shared_inputs = {k: v for k, v in shared_inputs.items() if len(v) > 1}

    return shared_inputs

def identify_end_of_branches(node_info):
    """Identify nodes that act as the end of branches by feeding into Concat operations."""
    end_of_branch_nodes = {}
    all_concat_inputs = []
    set_of_op_over_br = []

    # Identify all inputs to Concat operations
    for nidx, node in enumerate(node_info):
        if node["op_type"] == "Concat" or node["op_type"] == "Add":
            concat_inputs = set(node["inputs"])
            all_concat_inputs.append([node["node_name"], concat_inputs])
            set_of_op_over_br.append(nidx)

    #print(all_concat_inputs)  # Debug print to check the concat inputs

    # Any node producing an output that is used as a Concat input is an end of a branch
    for node in node_info:
        output_name = node["outputs"][0]
        for concat_name, concat_inputs in all_concat_inputs:
            if output_name in concat_inputs:
                end_of_branch_nodes[concat_name] = concat_inputs

    # Build a new dictionary with index lists for each concat node
    updated_end_of_branch_nodes = {}
    for bidx, (concat_name, concat_inputs) in enumerate(end_of_branch_nodes.items()):
        out_list = []
        for each_in in concat_inputs:
            for idx, node in enumerate(node_info):
                if each_in == node["outputs"][0]:
                    out_list.append(idx)
        updated_end_of_branch_nodes[concat_name] = out_list

    return set_of_op_over_br, updated_end_of_branch_nodes


def identify_branches(node_info, shared_inputs, end_of_branch_nodes, set_of_op_over_br):
    total_sets_of_branches = []
    set_of_op_goto_br = []



    def build_branch_structure(idx_ops, start_idx, end_idx, shar_input, merge_end, br_set_idx):
        return {
            "br_set_idx": br_set_idx,
            "shared_inputs_from": shar_input,
            "merged_by": merge_end,
            "br_num": idx_ops,
            "operations": list(range(start_idx, end_idx + 1)),
            "sub_branches": []  # This will now be a list of dicts
        }

    def build_sub_branch_structure(start_idx, end_idx, br_set_idx, br_id):
        return {
            "br_set_idx": br_set_idx,
            "br_id": br_id,
            "operations": list(range(start_idx, end_idx + 1))
        }

    #print(shared_inputs)
    for _, idx_ops in shared_inputs.items():
        #print("idx_ops:", idx_ops)
        start_idx = idx_ops[0]

        # Find the node providing input to the first op in this branch
        for i, n in enumerate(node_info):
            if node_info[start_idx]["inputs"][0] == n["outputs"][0]:
                set_of_op_goto_br.append(i)
                break

        # Check if this belongs to an existing branch set
        check_sub = False
        sub_in = -1
        for tidx, tobr in enumerate(total_sets_of_branches):
            if start_idx in tobr["operations"]:
                check_sub = True
                sub_in = tidx
                break

        sub_end = []
        if check_sub:
            for _, end_list in end_of_branch_nodes.items():
                if end_list[0] in total_sets_of_branches[sub_in]["operations"]:
                    sub_end = end_list
                    break

            for i, start in enumerate(idx_ops):
                tmp = start + 1
                end_idx = 0
                while True:
                    if tmp == max(sub_end):
                        end_idx = tmp
                        break
                    elif tmp in sub_end:
                        end_idx = tmp
                        break
                    tmp += 1

                sub_br = build_sub_branch_structure(start, end_idx, sub_in, i + 1)
                total_sets_of_branches[sub_in]["sub_branches"].append(sub_br)

        else:
            tmp = start_idx + 1
            end_idx = 0
            while True:
                for _, end_list in end_of_branch_nodes.items():
                    if tmp in end_list:
                        end_idx = max(end_list)
                        break
                if end_idx:
                    break
                tmp += 1

            #print(f"set_of_op_goto_br: {set_of_op_goto_br}")
            #print(f"set_of_op_over_br: {set_of_op_over_br}")
            shared_from = set_of_op_goto_br[-1] if set_of_op_goto_br else start_idx
            if set_of_op_goto_br and len(set_of_op_goto_br) <= len(set_of_op_over_br):
                merged_by = set_of_op_over_br[len(set_of_op_goto_br) - 1]
            else:
                merged_by = end_idx

            new_branch = build_branch_structure(idx_ops, start_idx, end_idx, shared_from, merged_by, len(total_sets_of_branches))
            #print(f"new_branch:{new_branch}")
            total_sets_of_branches.append(new_branch)

    return set_of_op_goto_br, total_sets_of_branches




# ---------- graph utils ----------
def _build_graph(node_info):
    prod = {}
    for i, n in enumerate(node_info):
        for t in n.get("outputs", []):
            prod[t] = i

    preds = [[] for _ in node_info]
    succs = [[] for _ in node_info]
    for j, n in enumerate(node_info):
        for t in n.get("inputs", []):
            if t in prod:
                i = prod[t]
                preds[j].append(i)
                succs[i].append(j)
    return preds, succs

def _topo_order(preds, succs):
    indeg = [len(p) for p in preds]
    q = deque([i for i, d in enumerate(indeg) if d == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(preds):
        raise RuntimeError("Graph has cycles")
    rank = [0] * len(order)
    for r, i in enumerate(order):
        rank[i] = r
    return order, rank

def _reachable_from(u, succs, limit=None):
    """Nodes reachable from u; if limit is set, stay inside that set."""
    vis = set([u])
    q = deque([u])
    while q:
        x = q.popleft()
        for y in succs[x]:
            if limit is not None and y not in limit:
                continue
            if y not in vis:
                vis.add(y); q.append(y)
    return vis

def _reaches(v, preds, limit=None):
    """Nodes that can reach v; if limit is set, stay inside that set."""
    vis = set([v])
    q = deque([v])
    while q:
        x = q.popleft()
        for y in preds[x]:
            if limit is not None and y not in limit:
                continue
            if y not in vis:
                vis.add(y); q.append(y)
    return vis

def _first_local_merge(split, preds, succs, topo_rank, region):
    """
    First node after `split` inside `region` that all branch heads reach.
    Bitmask propagation limited to `region`.
    """
    heads = [h for h in succs[split] if h in region]
    if len(heads) < 2:
        return None

    n = len(succs)
    head_to_bit = {h: i for i, h in enumerate(heads)}
    full = (1 << len(heads)) - 1

    mask = [0] * n
    for h, bit in head_to_bit.items():
        mask[h] = 1 << bit

    # process in topological order restricted to region
    order = sorted(region, key=lambda i: topo_rank[i])
    for v in order:
        cur = mask[v]
        for w in succs[v]:
            if w not in region:
                continue
            nm = mask[w] | cur
            if nm != mask[w]:
                mask[w] = nm

    # earliest node after split whose mask == full
    cands = [v for v in region if mask[v] == full and topo_rank[v] > topo_rank[split]]
    if not cands:
        return None
    return min(cands, key=lambda i: topo_rank[i])

def _region_between(split, merge, preds, succs):
    """Nodes strictly between split and merge (on some path split→…→merge)."""
    fwd = _reachable_from(split, succs)
    back = _reaches(merge, preds)
    return sorted((fwd & back) - {split, merge})

# ---------- recursive branch discovery ----------
def _collect_branchsets_in_region(region_nodes, preds, succs, topo_rank):
    """
    Find all branch-sets inside a region (recursively).
    region_nodes excludes the parent split & merge.
    """
    region = set(region_nodes)
    result = []
    seen_pairs = set()

    # scan possible splits in this region in topo order
    for u in sorted(region, key=lambda i: topo_rank[i]):
        # must have at least 2 successors inside region
        heads = [h for h in succs[u] if h in region]
        if len(heads) < 2:
            continue

        # local merge must also be inside this region
        m = _first_local_merge(u, preds, succs, topo_rank, region=region)
        if m is None or m not in region:
            continue

        key = (u, m)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        ops = _region_between(u, m, preds, succs)

        # compute actual branch heads that can reach m within this local span
        regset = set(ops) | {u, m}
        br_nums = []
        for h in succs[u]:
            if h not in regset:
                continue
            # reachability to m limited to this local span
            q = deque([h]); vis = {u}
            ok = False
            while q and not ok:
                x = q.popleft()
                if x == m:
                    ok = True; break
                for y in succs[x]:
                    if y in vis or y not in regset:
                        continue
                    vis.add(y); q.append(y)
            if ok:
                br_nums.append(h)

        # recurse for inner sub-branches
        sub = _collect_branchsets_in_region(ops, preds, succs, topo_rank)

        result.append({
            "br_set_idx": len(result),  # local index within this level
            "shared_inputs_from": u,
            "merged_by": m,
            "br_num": sorted(br_nums),
            "operations": ops,          # strictly inside [u, m)
            "sub_branches": sub
        })

    return result

def identify_branches_hierarchical(node_info):
    """
    Top-level branch-sets across the whole graph AND their sub-branches.
    Each branch-set's 'sub_branches' recursively lists nested branch-sets.
    """
    preds, succs = _build_graph(node_info)
    _, rank = _topo_order(preds, succs)

    n = len(node_info)
    total_sets = []
    seen_pairs = set()

    for u in range(n):
        if len(succs[u]) < 2:
            continue
        # search domain = descendants of u (keeps merge local)
        domain = _reachable_from(u, succs)
        m = _first_local_merge(u, preds, succs, rank, region=domain)
        if m is None:
            continue

        key = (u, m)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        ops = _region_between(u, m, preds, succs)

        # branch heads that really reach m within this span
        regset = set(ops) | {u, m}
        br_nums = []
        for h in succs[u]:
            if h not in regset:
                continue
            q = deque([h]); vis = {u}
            ok = False
            while q and not ok:
                x = q.popleft()
                if x == m:
                    ok = True; break
                for y in succs[x]:
                    if y in vis or y not in regset:
                        continue
                    vis.add(y); q.append(y)
            if ok:
                br_nums.append(h)

        # recursive discovery inside this region
        sub = _collect_branchsets_in_region(ops, preds, succs, rank)

        total_sets.append({
            "br_set_idx": len(total_sets),  # global index at top level
            "shared_inputs_from": u,
            "merged_by": m,                 # local concat/merge
            "br_num": sorted(br_nums),
            "operations": ops,
            "sub_branches": sub
        })

    return total_sets


def find_op_in_br(op, total_sets_of_branches, node_info=None):
    
    #print("op: ", op)
    for branch_set in total_sets_of_branches:
        #print("branch_set: ", branch_set)
        op_list = branch_set['operations']
        br_num = branch_set['br_num']
       # print(f"br_num:{br_num}")
        if node_info and op == branch_set['shared_inputs_from']:
            node_info[op]['br_label'] = "set_"+str(branch_set['br_set_idx'])+"_share_input"
        elif node_info and op == branch_set['merged_by']:
            node_info[op]['br_label'] = "set_"+str(branch_set['br_set_idx'])+"_merge_output"

        elif op in op_list:

            lower = max((b for b in br_num if b <= op), default=None)
            upper = min((b for b in br_num if b > op), default=None)
            if lower == br_num[-1]:
                upper = op_list[-1]
            elif upper > op:
                upper= upper-1

            #print([lower, upper])
            br_range = [b for b in op_list if lower <= b <= upper]
            #print(f"br_num range for op={op}: {br_range}")
            
            if node_info:
                if op == lower:
                    if op == upper:
                        node_info[op]['br_label'] = "set_"+str(branch_set['br_set_idx'])+"_br_"+str(br_num.index(lower)+1)+"_1op"
                    else:
                        node_info[op]['br_label'] = "set_"+str(branch_set['br_set_idx'])+"_br_"+str(br_num.index(lower)+1)+"_start"
                elif op == upper and op != lower:
                    node_info[op]['br_label'] = "set_"+str(branch_set['br_set_idx'])+"_br_"+str(br_num.index(lower)+1)+"_end"
                else:
                    node_info[op]['br_label'] = "set_"+str(branch_set['br_set_idx'])+"_br_"+str(br_num.index(lower)+1)+"_mid"

            return br_num.index(lower)+1, br_range, len(br_num)
        else:
            #print("branch_set: ", branch_set['br_set_idx'])
            if branch_set['br_set_idx'] < len(total_sets_of_branches)-1:
                continue

    return None, None, None



def calculate_memory_usage(inferred_model, node_info, op_order, total_sets_of_branches):
    ifm_sizes = []
    ofm_sizes = []
    buffer_sizes = []
    tensor_shapes = {i.name: get_tensor_shapes(i) for i in inferred_model.graph.input}
    data_usage=[]
    # Initial ifm, ofm and buffer size
    num = len(op_order)  # Number of operations to iterate over

    ifm_sizes = [0] * num
    ofm_sizes = [0] * num
    buffer_sizes = [0] * num

    tensor_list=[]
    # Calculate IFM and OFM sizes
    for idx in op_order:
        node = node_info[idx]

        # Calculate input memory size (IFM)
        ifm_size = 0
        for input_name in node["inputs"]:
            if input_name in tensor_shapes:
                size = calculate_tensor_size(tensor_shapes[input_name])
                ifm_size += size
                if input_name not in tensor_list:
                    data_entry = {
                        'name': input_name,
                        'index': len(tensor_list),
                        'from': [],
                        'to': [idx],
                        'tensor_shape': tensor_shapes[input_name],
                        'size': size
                    }
                    data_usage.append(data_entry)
                    tensor_list.append(input_name)
                    node['data_in'].append(data_entry['index'])
                else:
                    t_idx = tensor_list.index(input_name)
                    data_usage[t_idx]['to'].append(idx)
                    node['data_in'].append(t_idx)


        # Calculate output memory size (OFM) and update tensor_shapes
        ofm_size = 0
        for output_name in node["outputs"]:
            output_shape = node["output_shapes"][node["outputs"].index(output_name)]
            tensor_shapes[output_name] = output_shape
            if output_shape:
                size = calculate_tensor_size(output_shape)
                ofm_size += size

                if output_name not in tensor_list:
                    data_entry = {
                        'name': output_name,
                        'index': len(tensor_list),
                        'from': [idx],
                        'to': [],
                        'tensor_shape': tensor_shapes[output_name],
                        'size': size
                    }
                    data_usage.append(data_entry)
                    tensor_list.append(output_name)
                    node['data_out'].append(data_entry['index'])
                else:
                    t_idx = tensor_list.index(output_name)
                    data_usage[t_idx]['from'].append(idx)
                    node['data_out'].append(t_idx)

        node_info[idx] = node
        # Append each size to its respective list

        ifm_sizes[idx] = ifm_size
        ofm_sizes[idx] = ofm_size

    # Adjust buffer sizes based on branch structure
    buf_order = op_order
    ofm_finished = 0
    ifm_retain = 0

    #print(f"buf_order:{buf_order}")
    for idx in buf_order:
        #print(f"idx: {idx}")
        br, op_list, br_cnt = find_op_in_br(idx, total_sets_of_branches, node_info)
        #print("br=", br)
        #print("op_list=", op_list)
        if br is None:  # not in branch, do not need buffer
            buffer_sizes[idx] = 0
            ofm_finished = 0
        else:    # in branch, need buffer
            buffer_sizes[idx] = 0
            if br == 1: #1st br, no need for ofm_finished
                ofm_finished = 0
                if idx == op_list[0]:  # Start of branch
                    ifm_retain = ifm_sizes[idx]
                else:
                    buffer_sizes[idx] = ifm_retain + ofm_finished
                
                if idx == op_list[-1]:  # End of branch
                    ofm_finished = ofm_sizes[idx]
            elif br == br_cnt:   #last br in a set
                    buffer_sizes[idx] = ofm_finished
                
            else:

                if idx == op_list[0]:  # Start of branch
                    ifm_retain = ifm_sizes[idx]
                    buffer_sizes[idx] = ofm_finished
                else:
                    buffer_sizes[idx] = ifm_retain + ofm_finished

                if idx == op_list[-1]:  # End of branch
                    ofm_finished += ofm_sizes[idx]
            

    return ifm_sizes, ofm_sizes, buffer_sizes, data_usage, node_info


def plot_stacked_memory_usage(ifm_sizes, ofm_sizes, buffer_sizes, op_order, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = range(len(op_order))

    # Convert sizes to KB for readability
    ifm_sizes_kb = [m / 1024 for m in ifm_sizes]
    ofm_sizes_kb = [m / 1024 for m in ofm_sizes]
    buffer_sizes_kb = [m / 1024 for m in buffer_sizes]

    # Plot stacked bars at specific positions in op_order
    for i, idx in enumerate(op_order):
        ax.bar(i, ifm_sizes_kb[idx], label='IFM Size (KB)' if i == 0 else "", color="steelblue")
        ax.bar(i, ofm_sizes_kb[idx], bottom=ifm_sizes_kb[idx], label='OFM Size (KB)' if i == 0 else "", color="darkseagreen")
        ax.bar(i, buffer_sizes_kb[idx], bottom=ifm_sizes_kb[idx] + ofm_sizes_kb[idx], label='Buffer Size (KB)' if i == 0 else "", color="gold")

    # Set Y-axis limit if needed
    y_upper = (max(ifm_sizes_kb) + max(ofm_sizes_kb) + max(buffer_sizes_kb)) * 1.1
    ax.set_ylim(0, y_upper)

    # Set x-axis ticks to match op_order and use custom labels
    ax.set_xticks(pos)
    ax.set_xticklabels([str(x) for x in op_order])

    # Adjust x-axis limit to fit the range of op_order
    ax.set_xlim(min(op_order) - 1, max(op_order) + 3)

    # Labels and title
    ax.set_ylabel("Memory Usage (KB)")
    ax.set_title(f"Memory Usage per Operation (Stacked) - {model_name}")
    ax.set_xlabel("Operation Index")
    ax.legend()

    # Save the figure
    plt.savefig(f"{model_name}_mem_VM" + str(AVAILABLE_VM//1024)+".png", format='png', dpi=300)
    #plt.show()
    plt.close(fig)


def model_tracing(onnx_path, onnx_name):
    onnx_model = onnx.load(os.path.join(onnx_path, onnx_name + ".onnx"))
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    g = inferred_model.graph
    onnx.save(inferred_model, onnx_path+onnx_name+"_inferred.onnx")

    #Total_nodes = len(g.node)
    mem_constraint = AVAILABLE_VM 
    inplace_op=['Relu', 'Softmax','MaxPool', 'GlobalAveragePool', 'Squeeze', 'Add', 'Mul', 'Reshape',  'BatchNormalization', 'Sigmoid', 'Dropout','LRN', 'DequantizeLinear', 'QuantizeLinear', 'LeakyRelu','Split', 'Slice']
    ad_list={}  # {start node:[end node list]}
    node_list=[] # include nodes' number (for each op.)
    op_type=[] # include nodes' type (for each op.)
    data_usage=[]  # {data': name/idx, 'start': op., 'end': [op.], 'size': multiplied by dims}
    node_status=[] # {'name': op. idx, 'input':[data idx list], 'output':[data idx list], 'target_mem': peak_mem_size_plain, 'peak_mem':peak_mem_size_cur}
    buff=[None] # recording which data stored in buffer during buffer allocation
    buf_num_per_op=[] # the number of used buffers per op. 
    mem_stacked=[]
    mem_IOB=[]
    d_cnt=1
    # global total_MAC_gap, max_MAC_gap

    pp = PrettyPrinter(width=150)

    # Load and analyze the model
    input_info, output_info = get_io_info(inferred_model)
    #print("Model Inputs:", input_info)
    #print("Model Outputs:", output_info)

    # Get node I/O information and total node count
    node_info, total_nodes = get_node_io_info(inferred_model)
    #print(total_nodes)

     

    #print(op_order)
    shared_inputs = identify_shared_inputs(node_info)
    #print("shared_inputs: ", shared_inputs)

    set_of_op_over_br, end_of_branch_nodes = identify_end_of_branches(node_info)
    #print("end_of_branch_nodes: ", end_of_branch_nodes)
    #print("set_of_op_over_br: ", set_of_op_over_br)
    #set_of_op_goto_br, total_sets_of_branches = identify_branches(node_info, shared_inputs, end_of_branch_nodes, set_of_op_over_br)
    total_sets_of_branches = identify_branches_hierarchical(node_info)
    #print("set_of_op_goto_br: ", set_of_op_goto_br)
    #print("set_of_op_over_br: ", set_of_op_over_br)
    #print("Total Sets of Branches:", total_sets_of_branches)

    #op_order=[]
    #for i in range(0,total_nodes):
    #    op_order.append(i)
    op_order = list(range(total_nodes))
    #op_order = [0,1,14,15,16,8,9,10,11,12,13,4,5,6,7,2,3,17]
    #print(op_order)
    all_node_orders=[]
    all_node_orders.append(op_order)

    # Calculate memory usage  ->  affected by op. orders
    #mem_usage_all_orders=[]
    mem_usage = {}
    which_over_mem = []
    op_order_by_peak = op_order

    #for oridx, op_order in enumerate(all_node_orders):
    ifm_sizes, ofm_sizes, buffer_sizes, data_usage, node_info = calculate_memory_usage(inferred_model, node_info, op_order, total_sets_of_branches)
    mem_usage['ifm'] = ifm_sizes
    mem_usage['ofm'] = ofm_sizes
    mem_usage['buf'] = buffer_sizes
    total_per_op = []
    for idx in range(len(ifm_sizes)):
        total_per_op.append(ifm_sizes[idx] + ofm_sizes[idx] + buffer_sizes[idx])

    mem_usage['total'] = total_per_op
    mem_usage['op_order'] = op_order
    mem_usage['peak'] = max(total_per_op)
    mem_usage['peak_idx'] = total_per_op.index(mem_usage['peak'])
    
    for tidx, tot in enumerate(total_per_op):
        if tot > AVAILABLE_VM:
            which_over_mem.append(tidx) 


    which_over_mem = sorted(which_over_mem, key=lambda idx: total_per_op[idx], reverse=True)
    op_order_by_peak = sorted(op_order_by_peak, key=lambda idx: total_per_op[idx], reverse=True)

    mem_usage['over_mem_idx'] = which_over_mem

    #print("which_over_mem: ", which_over_mem)
    #print("op_order_by_peak: ", op_order_by_peak)
    #mem_usage_all_orders.append(mem_usage)

        #plot mem usage in different order
    model_name = onnx_path+onnx_name + '_order_0' 
        
    if EXPORT_FILE:
        plot_stacked_memory_usage(ifm_sizes, ofm_sizes, buffer_sizes, op_order, model_name)

        #pp.pprint(data_usage)

    #for midx, mu in enumerate(mem_usage_all_orders):
        #print("peak mem in order ",midx, " : ", mu['peak'])

    # Write mem_usage_all_orders to a text file with detailed information
    # === Export node_info ===
    if EXPORT_FILE:
        with open(onnx_path + onnx_name + "_node_info_VM" + str(AVAILABLE_VM//1024)+".txt", "w") as f:
            for idx, node in enumerate(node_info):
                f.write(f"--- Node {idx} ---\n")
                f.write(f"Name: {node['node_name']}\n")
                f.write(f"Type: {node['op_type']}\n")
                f.write(f"Inputs: {node['inputs']}\n")
                f.write(f"Outputs: {node['outputs']}\n")
                f.write(f"Input Shapes: {node['input_shapes']}\n")
                f.write(f"Output Shapes: {node['output_shapes']}\n")
                f.write(f"Attributes: {node['attributes']}\n")
                f.write(f"Data In Index: {node['data_in']}\n")
                f.write(f"Data Out Index: {node['data_out']}\n")
                f.write(f"Branch Label: {node['br_label']}\n")
                f.write("\n")
        
    if EXPORT_FILE:
        # === Export data_usage ===
        with open(onnx_path + onnx_name + "_data_usage_VM" + str(AVAILABLE_VM//1024)+".txt", "w") as f:
            for data in data_usage:
                f.write(f"--- Tensor {data['index']} ---\n")
                f.write(f"Name: {data['name']}\n")
                f.write(f"From Ops: {data['from']}\n")
                f.write(f"To Ops: {data['to']}\n")
                f.write(f"Shape: {data['tensor_shape']}\n")
                f.write(f"Size (bytes): {data['size']}\n")
                f.write("\n")


        #for idx, order in enumerate(mem_usage_all_orders):
        output_file = str(onnx_path+onnx_name)+"_mem_usage.txt"
        with open(output_file, "w") as f:
            opnum = len(op_order)
            for oidx in range(opnum):
                f.write(f"Op {oidx}:")
                f.write(f"  IFM: {mem_usage['ifm'][oidx]}")
                f.write(f"  OFM: {mem_usage['ofm'][oidx]}")
                f.write(f"  Buf: {mem_usage['buf'][oidx]}")
                f.write(f"  Total: {mem_usage['total'][oidx]}\n")
            f.write(f"  Peak Memory Usage: Op {mem_usage['peak_idx']} : {mem_usage['peak']}\n")
            f.write(f"  Op which_over_mem_cons: {mem_usage['over_mem_idx']}\n")
            f.write("\n")  # Add a blank line between orders

    #print(f"Memory usage details saved to {output_file}")


    ###-------------------- 3 modes -------------------------###
    # our method    
    if ENABLE_PDQSEL_MODE:
        #print(f"TS MODE: pdq - our pdq selection \n")
        peak_node=which_over_mem
        dup_path=[]
    
    #peak_node = find_peak_node(mem_IOB,mem_constriant)

        if GOAL_PRIORITY =='bal':
            print(f"TS MODE: pdq - our pdq selection - bal \n")
            if peak_node:
                #peak_node = sort_peak_nodes_by_io_sum(peak_node, mem_IOB)
                #print("peak_node list: ", peak_node)
                # Start time
                start_time = datetime.now()
                #dup_path, est_peak_per_path, q_list_per_path = 
                total_pdq_config, time_record, remaining_peaks = select_configurations(
                    node_list, peak_node, data_usage, node_info, mem_usage, AVAILABLE_VM, total_sets_of_branches, GOAL_PRIORITY)
                # End time
                end_time = datetime.now()
                
                duration = end_time - start_time
                #print("Select_configurations Elapsed time:", duration)
                #print("Seconds:", duration.total_seconds())


                if total_pdq_config:
                    if len(remaining_peaks)>0:
                        under_mem = False
                            
                    else:    
                        under_mem = True
                        for con in total_pdq_config:
                            if con['subgraph']:
                                under_mem = (con['after_peak_mem'] <= AVAILABLE_VM )

                            # for con in total_pdq_config:
                            #     if not con['dup_config']:
                            #         under_mem = False 
                                    #under_mem and con['dup_config']['under_mem']
                            #under_mem = all(con['after_peak_mem'] <= AVAILABLE_VM for con in total_pdq_config)
                else:
                    under_mem = False


            else:
                duration=0
                time_record=None
                total_pdq_config = None
                remaining_peaks = None
                #dup_path = None
                #est_peak_per_path = None
                #q_list_per_path = None
                under_mem = True


        if GOAL_PRIORITY == 'mem' and op_order_by_peak:
            print(f"TS MODE: pdq - our pdq selection - mem \n")


            start_time = datetime.now()
                #dup_path, est_peak_per_path, q_list_per_path = 
            total_pdq_config, time_record, remaining_peaks = select_configurations(
                node_list, op_order_by_peak, data_usage, node_info, mem_usage, AVAILABLE_VM, total_sets_of_branches, GOAL_PRIORITY)
                # End time
            end_time = datetime.now()
                
            duration = end_time - start_time

            under_mem = False
            peak_mem = 0
            if total_pdq_config:
                for saved in total_pdq_config:
                    if saved['after_peak_mem'] > peak_mem:
                        peak_mem = saved['after_peak_mem']

                if remaining_peaks:
                    if mem_usage['total'][remaining_peaks[0]] > peak_mem:
                        peak_mem = mem_usage['total'][remaining_peaks[0]]

                
            else:
                if remaining_peaks:
                    if mem_usage['total'][remaining_peaks[0]] > peak_mem:
                        peak_mem = mem_usage['total'][remaining_peaks[0]]
                else:
                    peak_mem = mem_usage['total'][op_order_by_peak[0]]

            
            if peak_mem > AVAILABLE_VM:
                under_mem = False
            else: 
                under_mem = True

        


        total_base_macs=0
        total_weight_access = 0
        total_increased_weight_access = 0
            
        for node in node_info:
            if node["op_type"] == "Conv":
                base_macs,_ = compute_conv_macs_from_node(node,1)
                #print(f"base_macs: {base_macs}, q = 1")
                total_base_macs += base_macs
                weight_access = compute_conv_access_from_node(node,1)
                #print(f"weight_access: {weight_access}, q = 1")
                total_weight_access += weight_access

        #print(f"total_base_macs: {total_base_macs}")
        #print(f"total_weight_access: {total_weight_access}")

        op_in_dup =[]
        total_split_macs_gap = 0
        if total_pdq_config : #and under_mem:
            total_split_macs_gap = 0
            for cf in total_pdq_config:
                subgraph = cf['subgraph']
                config = cf['dup_config']
                if subgraph:
                    for op in subgraph:
                        if node_info[op]["op_type"] == "Conv":
                            base_macs, split_macs = compute_conv_macs_from_node(node_info[op], config['q'])
                            #print(f"base_macs: {base_macs}, split_macs: {split_macs}, q = {config['q']}")
                            #split_macs = compute_conv_macs_with_split(node_info[op], config['q'])
                            # total_base_macs += base_macs
                            total_split_macs_gap += (split_macs - base_macs)
                            increased_weight_access = compute_conv_access_from_node(node_info[op],config['q'])
                            #print(f"increased_weight_access: {increased_weight_access}, q = {config['q']}")
                            total_increased_weight_access += increased_weight_access

                        op_in_dup.append(op)

            #print(f"total_split_macs_gap: {total_split_macs_gap}")
            #print(f"[MACs] increased percentage: {total_split_macs_gap/total_base_macs}\n")
            #print(f"total_increased_weight_access: {total_increased_weight_access}")
            #print(f"[Access] increased percentage: {total_increased_weight_access/total_weight_access}\n")
            ori_mac = total_base_macs
            plus_mac = total_split_macs_gap
            ori_access = total_weight_access
            plus_access = total_increased_weight_access

            gap_macs = total_split_macs_gap/total_base_macs
            access_gap = total_increased_weight_access/total_weight_access
                # total_MAC_gap += gap_macs
                # if max_MAC_gap < gap_macs:
                #     max_MAC_gap = gap_macs

        else:
            #print(f"TS does not help")
            ori_mac = total_base_macs
            plus_mac = 0
            ori_access = total_weight_access
            plus_access = 0
            gap_macs = 0
            access_gap = 0
            op_in_dup = []
            
        #print(f"op_in_dup: {op_in_dup}")
        max_mem_without_dup = 0
        for n in range(len(node_info)): 
            if n not in op_in_dup:
                if mem_usage['total'][n] > max_mem_without_dup:
                    max_mem_without_dup = mem_usage['total'][n]


    #baseline: tinyts
    elif ENABLE_MICROGRAPH_MODE:
        print(f"TS MODE: tinyts - micrograph-based \n")
        #print("[MODE] Using micrograph-based analysis")
        peak_node=which_over_mem
        time_record={"multi":[], "single":[]}
        # Start time
        inner_time=0
        start_time = datetime.now()
        micrographs = split_into_micrographs(node_info, unsplittable_ops, merging_ops, data_usage)
        mid_time = datetime.now()
        
        #print("micrographs: ", micrographs)


        #ori_peak_per_micro = [max((mem_usage['total'][op] for op in micro), default=0) for micro in micrographs]
        micrographs.sort(key=lambda micro: max((mem_usage['total'][op] for op in micro), default=0), reverse=True)
        #print("ori_peak_per_micro: ", ori_peak_per_micro)
        #print("sorted micrographs: ", micrographs)
        
        set_of_op_start_br = []
        set_of_op_end_br = []

        #print("total_sets_of_branches: ", total_sets_of_branches)
        for setbr in total_sets_of_branches:
            set_of_op_start_br.append(setbr['shared_inputs_from'])
            set_of_op_end_br.append(setbr['merged_by'])

        split_duration = mid_time-start_time
        duplicated_per_micro = []
        split_ts_per_micro = []
        peak_mem_per_micro = []

        lifetime_by_mode = {}
        MACs_num_per_micro = []
        MACs_num_per_micro_ori = []
        access_per_micro = []
        incrased_access_per_micro = []

        op_in_micro = set()  

        unsplittable_check = [1]*len(node_info)


        for m_id, micro in enumerate(micrographs):
            mid2_time = datetime.now()
        
            #print(f"Micro-Graph {m_id}: {micro}")
            op_in_micro.update(micro)

            check_pn = False
            for op in micro:
                node_info[op]['micro_id'] = m_id
                unsplittable_check[op] = 0

            if GOAL_PRIORITY == 'bal':
                if op in peak_node:
                    check_pn = True 

            if GOAL_PRIORITY == 'mem':                
                current_peak = max((mem_usage['total'][op] for op in micro), default=0)
                if peak_mem_per_micro:
                    check_pn = True
                    for (best_peak, best_mode, best_sp) in peak_mem_per_micro:
                        #print(f"current_peak: {current_peak}, pre_saved: {best_peak} ")
                        if best_peak >= current_peak:
                            #print("!! Exit MICROGRAPH-MEM selection !!  Peak memory cannot be reduced further since the current saved MICROGRAPH-TS remains above the next micro.")
                            check_pn = False
                else:
                    check_pn = True
                

                #print(f"  Op {op}: Total Mem = {mem_usage['total'][op]}  Type: {node_info[op]['op_type']}")
            #peak_nodes = sorted(micro, key=lambda x: mem_usage['total'][x], reverse=True)
            #for op in peak_nodes:
                #print(f"  Op {op}: Total Mem = {mem_usage['total'][op]}  Type: {}")
            
            if check_pn:
                best_peak_per_sp = 0
                best_mode_per_sp = ''
                best_sp = 2
                sp_record ={}
                for sp_height in SP_HEIGHT_LIST:
                #duplicated_ops, split_tensors = estimate_operator_duplication(micro, node_info, split_height=1)
                    duplicated_ops, split_tensors = estimate_operator_duplication(micro, node_info, sp_height)
                    
                    #for op_idx, num_copies in duplicated_ops:
                        #print(f"    -> Need to duplicate Op {op_idx} x{num_copies}")

                    expanded_ops = build_expanded_ops(micro, duplicated_ops, split_tensors, node_info)

                    # End time
                    end_time = datetime.now()
                    inner_duration = (end_time- mid2_time)
                    inner_time = inner_duration.total_seconds()
                    best_mode = ''
                    best_peak = 0
                    for mode in ['dfs', 'bfs']: #, 'bfs'
                        sched = schedule_ops(expanded_ops, mode)
                        #print(f"mode: {mode}")
                        #print(sched)
                        life = get_split_tensor_lifetime(sched, data_usage)
                        #print(f"life :{life}")
                        #life = get_tensor_lifetime(sched, expanded_ops, mode)
                        lifetime_by_mode[(m_id, mode)] = life
                        #print(f"\n  [{mode.upper()} Schedule] Lifetime:\n")
                        #for t, (start, end, size) in life.items():
                            #print(f"    Tensor {t}: {start} → {end} ; size:{size}\n")
                        
                        opname_to_step = build_opname_to_step(sched)
                        #print(opname_to_step)
                        #peak_mem, timeline = estimate_peak_memory_from_lifetime(life, data_usage, opname_to_step)
                        peak_bin, tensor_offsets,retain_from_br_in = bin_packing_memory_allocation(life, data_usage, node_info, opname_to_step, set_of_op_start_br, set_of_op_end_br)

                        if retain_from_br_in >0:
                            br_idx = 0
                            #print(sched[0])
                            (opidx, tin, tout, _) = sched[0]
                            first_op, copies = opidx.split("_") 
                            for setbr in total_sets_of_branches:
                                if int(first_op) in setbr['operations']: 
                                    heads = sorted(setbr['br_num'])
                                    m = setbr['merged_by']
                                    break
                            
                            op = int(first_op)

                            # only consider ops between split and merge
                            if not (setbr['shared_inputs_from'] < op < m):
                                br_idx = None  # outside this branch set (or shared nodes). Handle as you wish.
                            else:
                                # build half-open intervals [head_i, head_{i+1}) and last [head_last, merged_by)
                                ends = heads[1:] + [m]
                                br_idx = None
                                for i, (lo, hi) in enumerate(zip(heads, ends)):
                                    if lo <= op < hi:
                                        br_idx = i
                                        break
                            
                            #print("br_idx: ", br_idx)

                            if br_idx < (len(heads) - 1):
                                peak_bin += retain_from_br_in

                        if best_peak ==0 :
                            best_peak = peak_bin
                            best_mode = mode
                        else:
                            if best_peak > peak_bin:
                                best_peak = peak_bin
                                best_mode = mode


                        #if 
                    
                    if best_peak_per_sp == 0:
                        best_peak_per_sp = best_peak
                        best_mode_per_sp = best_mode
                        best_sp = sp_height
                    else:
                        if best_peak_per_sp > best_peak:
                            best_peak_per_sp = best_peak
                            best_mode_per_sp = best_mode
                            best_sp = sp_height
                    
                
                peak_mem_per_micro.append((best_peak_per_sp, best_mode_per_sp, best_sp))
                duplicated_ops, split_tensors = estimate_operator_duplication(micro, node_info, best_sp)
                duplicated_per_micro.append(duplicated_ops)
                split_ts_per_micro.append(split_tensors)

                total_base_macs = 0
                total_split_macs_gap = 0
                total_weight_access = 0
                total_increased_weight_access = 0

                for op in micro:
                    if node_info[op]["op_type"] == "Conv":
                        #if node_info[op]["input_shapes"][0][2] == None:
                            #print(node_info[op])
                        #print((node_info[op]["input_shapes"][0][2], peak_mem_per_micro[-1][2]))

                        num_sp = (node_info[op]["input_shapes"][0][2] // peak_mem_per_micro[-1][2]) // node_info[op]["attributes"]['kernel_shape'][0]
                        
                        base_macs, split_macs = compute_conv_macs_from_node(node_info[op], num_sp)
                        #split_macs = compute_conv_macs_with_split(node_info[op], num_sp)
                        total_base_macs += base_macs
                        total_split_macs_gap += (split_macs-base_macs)
                        weight_access = compute_conv_access_from_node(node_info[op],1)
                        total_weight_access += weight_access
                        increased_weight_access = compute_conv_access_from_node(node_info[op],num_sp)
                        total_increased_weight_access += increased_weight_access

                #print(f"total_base_macs: {total_base_macs}")
                #print(f"total_split_macs: {total_split_macs}\n")

                
                MACs_num_per_micro_ori.append(total_base_macs)
                MACs_num_per_micro.append(total_split_macs_gap)
                access_per_micro.append(total_weight_access)
                incrased_access_per_micro.append(total_increased_weight_access)
                total_split_macs_gap = 0

            else:
                maxone_inmicro = 0
                total_base_macs = 0
                total_weight_access = 0
                duplicated_op_num = []
                split_ts_num = []
                for op in micro:
                    #print(f"  Op {idx}: Type = {node_info[idx]['op_type']}, Mem = {mem_usage['total'][idx]}")
                    if mem_usage['total'][op] > maxone_inmicro:
                        maxone_inmicro = mem_usage['total'][op]
                    if node_info[op]["op_type"] == "Conv":
                        base_macs,_ = compute_conv_macs_from_node(node_info[op],1)
                        total_base_macs += base_macs
                        weight_access = compute_conv_access_from_node(node_info[op],1)
                        total_weight_access += weight_access

                    duplicated_op_num.append((op, 1))
                    split_ts_num.append((op, 1))

                peak_mem_per_micro.append((maxone_inmicro, None, None))
                MACs_num_per_micro_ori.append(total_base_macs)
                MACs_num_per_micro.append(0)

                access_per_micro.append(total_weight_access)
                incrased_access_per_micro.append(0)

                duplicated_per_micro.append(duplicated_op_num)
                split_ts_per_micro.append(split_ts_num)

        
                
        #print("Unsplittable Operators:")
        maxone = 0
        for idx, uc in enumerate(unsplittable_check):
            #print(f"  Op {idx}: Type = {node_info[idx]['op_type']}, Mem = {mem_usage['total'][idx]}")
            if (uc == 1) and (0 <= idx < len(mem_usage['total'])):
                if mem_usage['total'][idx] > maxone:
                    maxone = mem_usage['total'][idx]

        #print(f" Peak memory of Unsplittable ops: {maxone}")
        inner_time += split_duration.total_seconds()
        total_macs_without_split = sum(MACs_num_per_micro_ori)
        total_macs_after_split = sum(MACs_num_per_micro)
        total_access_without_split = sum(access_per_micro)
        total_access_after_split = sum(incrased_access_per_micro)
        #print(f"total_macs_without_split: {total_macs_without_split}")
        #print(f"total_macs_gap_after_split: {total_macs_after_split}")
        #print(f"[MACs] increased percentage: {total_macs_after_split/total_macs_without_split}\n")
        #print(f"total_access_without_split: {total_access_without_split}")
        #print(f"total_access_gap_after_split: {total_access_after_split}")
        #print(f"[Access] increased percentage: {total_access_after_split/total_access_without_split}\n")
       
        ori_mac = total_macs_without_split
        plus_mac = total_macs_after_split
        ori_access = total_access_without_split
        plus_access = total_access_after_split


        #gap_macs = total_macs_after_split/total_macs_without_split
        #access_gap = total_access_after_split/total_access_without_split
        # if max_MAC_gap < gap_macs:
        #     max_MAC_gap = gap_macs
    
    elif ENABLE_TINYNAS_MODE:
        print(f"TS MODE: tinynas - patch-based \n")

        patch_evals = []
        tried_but_not_valid = []
        opinbr = []
        for brset in total_sets_of_branches:
            opinbr += brset['operations']
        
        tried_num = 0
        for end in range(total_nodes):
            if end == 0:
                continue
            patch_ops = list(range(0, end + 1))
            if any(node_info[op]['op_type'] in unsplittable_ops for op in patch_ops):
                break  # skip if any op is unsplittable

            for q in [1, 2, 3, 4]:
                tried_num += 1
                mem_ops = []
                valid = True
                retain_from_patch_in = 0
                retain_from_patch_out = 0

                total_base_macs = 0
                total_split_macs_gap = 0
                total_weight_access = 0
                total_increased_weight_access = 0

                for op in patch_ops:
                    ishape = node_info[op]['input_shapes'][0]
                    oshape = node_info[op]['output_shapes'][0]

                    if not ishape or not oshape or ishape[2] is None or oshape[2] is None:
                        valid = False
                        break

                    ih, iw = ishape[2], ishape[3]
                    oh, ow = oshape[2], oshape[3]
                    ishape_patch = [ishape[0], ishape[1], ih // q, iw // q]
                    oshape_patch = [oshape[0], oshape[1], oh // q, ow // q]
                    input_size = calculate_tensor_size(ishape_patch)
                    output_size = calculate_tensor_size(oshape_patch)

                    if op == patch_ops[0]:
                        retain_from_patch_in = input_size
                    if op == patch_ops[-1]:
                        retain_from_patch_out = output_size

                    mem_ops.append(input_size + output_size + mem_usage['buf'][op])

                    if node_info[op]['op_type'] == "Conv":
                        # num_sp = q * q
                        base_macs, split_macs = compute_conv_macs_from_node(node_info[op], q)
                        total_base_macs += base_macs
                        total_split_macs_gap += (split_macs - base_macs)
                        total_weight_access += compute_conv_access_from_node(node_info[op], 1)
                        total_increased_weight_access += compute_conv_access_from_node(node_info[op], q, dim="2d")

                if not valid or not mem_ops:
                    continue

                retain_from_patch = 0
                if q > 1:
                    for a in range(q * q):
                        retain_a = (q * q - (a + 1)) * retain_from_patch_in + a * retain_from_patch_out
                        retain_from_patch = max(retain_from_patch, retain_a)
                    mem_ops = [m + retain_from_patch for m in mem_ops]

                total_mem = max(mem_ops)
                latency_gap = total_split_macs_gap + total_increased_weight_access  #check ratio


                ops_with_patch = []
                ops_without_patch = []
                maxone_unsp = 0
                for nidx in range(total_nodes):
                    if nidx not in range(0,end+1):
                        ops_without_patch.append(nidx)
                        if mem_usage['total'][nidx] > maxone_unsp:
                            maxone_unsp = mem_usage['total'][nidx]


                other_base_macs = 0
                other_weight_access = 0 
                
                
                for nidx in range(total_nodes):
                    if nidx not in range(0,end+1):
                        if node_info[nidx]["op_type"] == "Conv":
                            base_macs, split_macs = compute_conv_macs_from_node(node_info[nidx], 1)
                            other_base_macs+=base_macs
                            other_weight_access += compute_conv_access_from_node(node_info[nidx], 1)


                if total_mem <= AVAILABLE_VM:

                    if maxone_unsp <= AVAILABLE_VM:
                        patch_evals.append({
                            "start": 0,
                            "end": end,
                            "q": q,
                            "peak_mem_within_patch": total_mem,
                            "maxone_unsp": maxone_unsp,
                            "total_peak_mem": max(maxone_unsp, total_mem),
                            "latency_gap": latency_gap,
                            "mac_ori": total_base_macs,
                            "mac_gap": total_split_macs_gap,
                            "mac_other": other_base_macs,
                            "access_ori": total_weight_access,
                            "access_gap": total_increased_weight_access,
                            "access_other": other_weight_access,
                            "ops_with_patch": list(range(0,end+1)),
                            "ops_without_patch": ops_without_patch,
                            
                        })

                    else:
                        tried_but_not_valid.append({
                            "start": 0,
                            "end": end,
                            "q": q,
                            "peak_mem_within_patch": total_mem,
                            "maxone_unsp": maxone_unsp,
                            "total_peak_mem": max(maxone_unsp, total_mem),
                            "latency_gap": latency_gap,
                            "mac_ori": total_base_macs,
                            "mac_gap": total_split_macs_gap,
                            "mac_other": other_base_macs,
                            "access_ori": total_weight_access,
                            "access_gap": total_increased_weight_access,
                            "access_other": other_weight_access,
                            })

                else:
                    tried_but_not_valid.append({
                        "start": 0,
                        "end": end,
                        "q": q,
                        "peak_mem_within_patch": total_mem,
                        "maxone_unsp": maxone_unsp,
                        "total_peak_mem": max(maxone_unsp, total_mem),
                        "latency_gap": latency_gap,
                        "mac_ori": total_base_macs,
                        "mac_gap": total_split_macs_gap,
                        "mac_other": other_base_macs,
                        "access_ori": total_weight_access,
                        "access_gap": total_increased_weight_access,
                        "access_other": other_weight_access,
                        })


#ops_with_patch, ops_without_patch, maxone_unsp,
        # === Find best patch results
        if patch_evals:
            min_mem_result = min(patch_evals, key=lambda x: x['total_peak_mem'])
            min_latency_result = min(patch_evals, key=lambda x: x['mac_gap'])

        else:
            min_mem_result = None
            min_latency_result = None

            #print("\nBest memory-efficient patch:")
            #print(min_mem_result)

            #print("\nBest latency-efficient patch:")
            #print(min_latency_result)
        if tried_but_not_valid:
            min_tried_not_pass = min(tried_but_not_valid, key=lambda x: x['total_peak_mem'])
        else:
            min_tried_not_pass = None
        # === Output
        
        
        #print(f"tried_num:{tried_num}")
        
        

    else:
        total_macs = 0
        total_access = 0
        for nidx in range(total_nodes):
            if node_info[nidx]["op_type"] == "Conv":
                base_macs, split_macs = compute_conv_macs_from_node(node_info[nidx], 1)
                total_macs+=base_macs
                total_access += compute_conv_access_from_node(node_info[nidx], 1)


      

            

       
    total_latency = 0
    #= sum(con['after_peak_mem'] for con in total_pdq_config)
    if ENABLE_PDQSEL_MODE:
        return which_over_mem, mem_usage['peak'] ,total_latency, under_mem, total_pdq_config, duration, time_record, remaining_peaks, ori_mac, plus_mac, ori_access, plus_access, max_mem_without_dup
    elif ENABLE_MICROGRAPH_MODE:
        return which_over_mem, mem_usage['peak'] , duplicated_per_micro, split_ts_per_micro, micrographs, peak_mem_per_micro, inner_time, unsplittable_op_indices, maxone, ori_mac, plus_mac, ori_access, plus_access
    elif ENABLE_TINYNAS_MODE:
        return which_over_mem, mem_usage['peak'] , min_mem_result, min_latency_result, min_tried_not_pass
    else:
        return which_over_mem, mem_usage['peak'], total_macs, total_access
    #dup_path, est_peak_per_path, q_list_per_path
    #dup_path, est_peak_per_path, q_list_per_path



# === Test Entrypoint ===
def run_test(onnx_file_path):

    model_name = os.path.splitext(os.path.basename(onnx_file_path))[0]
    onnx_path = os.path.dirname(onnx_file_path) + '/'
    

    #full_start_time = datetime.now()
    full_start_time = time.time()

    if ENABLE_PDQSEL_MODE:
        which_over_mem, oripeak, total_latency, under_mem, total_pdq_config, duration, time_record, remaining_peaks, ori_mac, plus_mac, ori_access, plus_access, max_mem_without_dup = model_tracing(onnx_path, model_name)
    elif ENABLE_MICROGRAPH_MODE:
        which_over_mem, oripeak, duplicated_per_micro, split_ts_per_micro, micrographs, peak_mem_per_micro, inner_time, unsplittable_op_indices, maxone_unsp, ori_mac, plus_mac, ori_access, plus_access = model_tracing(onnx_path, model_name)
    elif ENABLE_TINYNAS_MODE:
        which_over_mem, oripeak, min_mem_result, min_latency_result, tried_but_not_valid = model_tracing(onnx_path, model_name)

    else:
        which_over_mem, oripeak, total_macs, total_access = model_tracing(onnx_path, model_name) 

    #full_end_time = datetime.now()
    full_end_time = time.time()
    
    #pp.pprint(total_pdq_config)

    full_duration = full_end_time - full_start_time
    #print("Full_Elapsed time:", full_duration)
    #print("Seconds:", full_duration.total_seconds())

    if ENABLE_PDQSEL_MODE:
        #print("total_latency: ", total_latency)
        #print("under_mem: ", under_mem)
        peak_after_dup = 0
        if total_pdq_config:
            
            for idx, entry in enumerate(total_pdq_config):
                if entry['subgraph'] and (entry['after_peak_mem'] > peak_after_dup):
                    peak_after_dup = entry['after_peak_mem']
        
        peak_for_all = max(max_mem_without_dup, peak_after_dup)

        if EXPORT_FILE:
            output_file = onnx_path+model_name+"_pdq_config_detail_VM"+str(AVAILABLE_VM//1024)+"_goal_"+GOAL_PRIORITY+".txt"

            with open(output_file, "w") as f:
                f.write(f"under_mem:{under_mem}\n")
                f.write(f"peak node list:{which_over_mem}\n"), 
                f.write(f"Full_duration:{full_duration}(s)\n")
                f.write(f"Peak memory for all: : {peak_for_all} bytes ({peak_for_all // 1024} KB)\n")
                
                if total_pdq_config:
                    f.write(f"Peak memory under dup: : {peak_after_dup} bytes ({peak_after_dup // 1024} KB)\n")
                    f.write(f"ori MAC: {ori_mac}\n") 
                    f.write(f"ori weight access: {ori_access}\n\n") 
                    f.write(f"Total MAC increased: {plus_mac}\n") 
                    f.write(f"Total weight access increased: {plus_access}\n\n")
                    f.write(f"The useful subgraphs idx: {[idx for idx, cf in enumerate(total_pdq_config) if cf['dup_config'] and cf['dup_config']['under_mem']]}\n")
                    f.write(f"remaining_peaks: {[rp for rp in remaining_peaks]}\n\n")

                    f.write("Total PDQ Configuration Details\n")
                    f.write("="*40 + "\n")
                    for idx, entry in enumerate(total_pdq_config):
                        f.write(f"Subgraph #{idx+1}\n")
                        f.write(f"  Subgraph Ops: {entry['subgraph']}\n")
                        f.write(f"  Original Peak Memory: {entry['ori_peak_mem']}\n")
                        f.write(f"  Optimized Peak Memory: {entry['after_peak_mem']}\n")
                        f.write(f"  Peak Operator Index: {entry['peak']}\n")
                        f.write("  Duplication Config:\n")
                        if entry['dup_config']:
                            dup = entry['dup_config']
                            f.write(f"    under memory constraint: {dup['under_mem']}\n")
                            f.write(f"    Selected d: {dup['d']}\n")
                            f.write(f"    d_candidate_num: {dup['d_candidate_num']}\n")
                            f.write(f"    Selected q: {dup['q']}\n")
                            f.write(f"    q_candidates: {dup['q_candidate']}\n")
                            f.write(f"    Total Memory Under q: {dup['total_mem_under_q']}\n")
                            f.write(f"    Final Peak Memory: {dup['peak_mem']}\n")
                        f.write("-"*40 + "\n")
                else:
                    f.write(f"Peak memory: : no need to dup, same as original one\n")

        
    elif ENABLE_MICROGRAPH_MODE and micrographs:
        peak_after_sp = 0
        for m_id, micro in enumerate(micrographs):
            micro_peak, micro_mode, micro_sp = peak_mem_per_micro[m_id]

            if micro_peak > peak_after_sp:
                peak_after_sp = micro_peak
        
        if maxone_unsp > peak_after_sp:
            peak_after_sp = maxone_unsp

        if EXPORT_FILE:
            output_file = onnx_path+model_name+"_micrograph_rep_"+str(AVAILABLE_VM//1024)+"_goal_"+GOAL_PRIORITY+".txt"
            with open(output_file, "w") as f:
                f.write(f"Full_duration:{full_duration}(s)\n")
                f.write(f"peak node list:{which_over_mem}\n"), 
                f.write(f"MICROGRAPH_duration:{inner_time}(s)\n")
                f.write(f"Peak memory: : {peak_after_sp} bytes ({peak_after_sp // 1024} KB)\n")
                f.write(f"ori MAC: {ori_mac}\n") 
                f.write(f"ori weight access: {ori_access}\n\n") 
                f.write(f"Total MAC increased: {plus_mac}\n") 
                f.write(f"Total weight access increased: {plus_access}\n\n")
                    
                for m_id, micro in enumerate(micrographs):
                    f.write(f"\nMicro-Graph {m_id}: {micro}\n")
                    f.write(f"peak_mem_per_micro: {peak_mem_per_micro[m_id]}\n")
                    #for op in micro:
                    #    node_info[op]['micro_id'] = m_id
                    #    ni = node_info[op]
                    #    f.write(f"  Op {op}: Type={ni['op_type']}, Mem={mem_usage['total'][op]}\n")
                    #    f.write(f"        InputShape={ni['input_shapes']}, OutputShape={ni['output_shapes']}\n")

                    
                    for(op_idx, num_copies),(_,split_ts) in zip(duplicated_per_micro[m_id], split_ts_per_micro[m_id]):
                        f.write(f"    -> Need to duplicate Op {op_idx} x{num_copies} for split input tensor num: {split_ts} \n")

                f.write("\nUnsplittable Operators:\n")
                f.write(f"{','.join(map(str, unsplittable_op_indices))}\n")

    elif ENABLE_TINYNAS_MODE:
        #which_over_mem, min_mem_result, min_latency_result


        if EXPORT_FILE:
            output_file = onnx_path + model_name + "_tinynas_repfor2_VM" + str(AVAILABLE_VM // 1024)+ ".txt"
            with open(output_file, "w") as f:
                f.write(f" Full_duration: {full_duration}(s)\n")
                f.write(f" Peak node list:{which_over_mem}\n"), 
                f.write("-" * 40 + "\n")
                
                if min_latency_result:

                    f.write(f" Patch selected by min latency/comp: \n")
                    f.write(f" Start Op: {min_latency_result['start']}, End Op: {min_latency_result['end']}\n")
                    f.write(f" Patch Split q: {min_latency_result['q']}\n")
                    f.write(f" >>> Total Peak Memory : {min_latency_result['total_peak_mem']} bytes ({min_latency_result['total_peak_mem'] // 1024} KB)\n")
                    f.write(f" Peak Memory Usage within Patch: {min_latency_result['peak_mem_within_patch']} bytes ({min_latency_result['peak_mem_within_patch'] // 1024} KB)\n")
                    f.write(f" Peak Memory Usage per layer: {min_latency_result['maxone_unsp']} bytes ({min_latency_result['maxone_unsp'] // 1024} KB)\n")
                    f.write(f" >>> full mac plus: {min_latency_result['mac_gap']}\n")
                    f.write(f" >>> full access plus : {min_latency_result['access_gap']}\n")
                    f.write(f" full mac ori: {min_latency_result['mac_ori'] + min_latency_result['mac_other']}\n")
                    f.write(f" full access ori : {min_latency_result['access_ori'] + min_latency_result['access_other']}\n")
                    
                    f.write(f" Within Patch: \n")
                    f.write(f" -- mac_ori : {min_latency_result['mac_ori']}\n")
                    f.write(f" -- mac_gap : {min_latency_result['mac_gap']}\n")
                    f.write(f" -- access_ori : {min_latency_result['access_ori']}\n")
                    f.write(f" -- access_gap : {min_latency_result['access_gap']}\n")
                
                if min_mem_result:

                    f.write("-" * 40 + "\n")
                    f.write(f" Patch selected by min memory: \n")
                    f.write(f" Start Op: {min_mem_result['start']}, End Op: {min_mem_result['end']}\n")
                    f.write(f" Patch Split q: {min_mem_result['q']}\n")
                    f.write(f" >>> Total Peak Memory : {min_mem_result['total_peak_mem']} bytes ({min_mem_result['total_peak_mem'] // 1024} KB)\n")
                    f.write(f" Peak Memory Usage within Patch: {min_mem_result['peak_mem_within_patch']} bytes ({min_mem_result['peak_mem_within_patch'] // 1024} KB)\n")
                    f.write(f" Peak Memory Usage per layer: {min_mem_result['maxone_unsp']} bytes ({min_mem_result['maxone_unsp'] // 1024} KB)\n")
                    f.write(f" >>> full mac plus: {min_latency_result['mac_gap']}\n")
                    f.write(f" >>> full access plus : {min_latency_result['access_gap']}\n")
                    f.write(f" full mac ori: {min_latency_result['mac_ori'] + min_latency_result['mac_other']}\n")
                    f.write(f" full access ori : {min_latency_result['access_ori'] + min_latency_result['access_other']}\n")
                    
                    f.write(f" Within Patch: \n")
                    f.write(f" -- mac_ori : {min_mem_result['mac_ori']}\n")
                    f.write(f" -- mac_gap : {min_mem_result['mac_gap']}\n")
                    f.write(f" -- access_ori : {min_mem_result['access_ori']}\n")
                    f.write(f" -- access_gap : {min_mem_result['access_gap']}\n")
                
                if tried_but_not_valid:

                    f.write("-" * 40 + "\n")
                    f.write(f" Patch selected by min memory: \n")
                    f.write(f" Start Op: {tried_but_not_valid['start']}, End Op: {tried_but_not_valid['end']}\n")
                    f.write(f" Patch Split q: {tried_but_not_valid['q']}\n")
                    f.write(f" >>> Total Peak Memory : {tried_but_not_valid['total_peak_mem']} bytes ({tried_but_not_valid['total_peak_mem'] // 1024} KB)\n")
                    f.write(f" Peak Memory Usage within Patch: {tried_but_not_valid['peak_mem_within_patch']} bytes ({tried_but_not_valid['peak_mem_within_patch'] // 1024} KB)\n")
                    f.write(f" Peak Memory Usage per layer: {tried_but_not_valid['maxone_unsp']} bytes ({tried_but_not_valid['maxone_unsp'] // 1024} KB)\n")
                    f.write(f" >>> full mac plus: {tried_but_not_valid['mac_gap']}\n")
                    f.write(f" >>> full access plus : {tried_but_not_valid['access_gap']}\n")
                    f.write(f" full mac ori: {tried_but_not_valid['mac_ori'] + tried_but_not_valid['mac_other']}\n")
                    f.write(f" full access ori : {tried_but_not_valid['access_ori'] + tried_but_not_valid['access_other']}\n")
                    
                    f.write(f" Within Patch: \n")
                    f.write(f" -- mac_ori : {tried_but_not_valid['mac_ori']}\n")
                    f.write(f" -- mac_gap : {tried_but_not_valid['mac_gap']}\n")
                    f.write(f" -- access_ori : {tried_but_not_valid['access_ori']}\n")
                    f.write(f" -- access_gap : {tried_but_not_valid['access_gap']}\n")


            

    if ENABLE_PDQSEL_MODE:
        return full_duration, under_mem, mac_count
    elif ENABLE_MICROGRAPH_MODE:
        return full_duration, peak_after_sp, mac_count
    elif ENABLE_TINYNAS_MODE:
        return full_duration, min_mem_result, min_latency_result
    else:
        return full_duration
    #.total_seconds()

    #if time_record:
        #output_file = onnx_path+model_name+"_time_record_detail_VM"+str(AVAILABLE_VM//1024)+".txt"

        #with open(output_file, "w") as f:
            #f.write(f"Full_duration:{full_duration.total_seconds()}(s)\n")
            #f.write(f"inner_duration:{duration.total_seconds()}(s)  # 0 means dup false\n")
            #f.write(f"time_record for multi-branch dxq search:\n")
            #for mt in time_record['multi']:
                #f.write(f"{mt.total_seconds()}(s)\n")
            #f.write(f"time_record for single-branch dxq search:\n")
            #for st in time_record['single']:
                #f.write(f"{st.total_seconds()}(s)\n")


if __name__ == "__main__":
    #test_model_path = "./sample_shuffle.onnx" #shuffle1,2 incept1,2,3
    total_time = 0
    max_time = 0
    # test_model_path = "./sample_incept1.onnx"
    # full_sec = run_test(test_model_path)

    # total_time += full_sec
    # avg_time = full_sec
    # max_time = full_sec
    pass_cons = 0
    total_MAC_gap = 0
    max_MAC_gap = 0
    mac_count = 0

    #test_model_path = "./inception_v3_x0_25_64.onnx"
    test_model_path = DIR_PATH + MODEL_NAME + ".onnx"
    #test_model_path = "./shufflenet_v2_x0_25.onnx"
    #test_model_path = "./shufflenet_v2_x0_5.onnx"
    #test_model_path = "./case_ok.onnx"


    if ENABLE_PDQSEL_MODE:
        full_sec, under_mem, mac_count = run_test(test_model_path)
        if under_mem:
            pass_cons += 1
    elif ENABLE_MICROGRAPH_MODE:
        full_sec, peak_after_sp, mac_count = run_test(test_model_path)
        if peak_after_sp <= AVAILABLE_VM:
            pass_cons += 1
    elif ENABLE_TINYNAS_MODE:
        full_sec, min_mem_result, min_latency_result = run_test(test_model_path)
        if min_mem_result:
            pass_cons += 1
        if min_latency_result:
            pass_cons += 1

    else:  # only tracing model to get memory usage info
        full_sec = run_test(test_model_path)

    print(f"pass_cons: {pass_cons}, total_percent: 1 \n")
    print(f"execution_time: {full_sec}\n")
   # print(f"mac_count: {mac_count}\n")

        
    # for i in range(16,41,1): 
    #     test_model_path = "./sample_"+str(i)+".onnx"
    # #for i in range(1,2,1):
    # #    test_model_path = "./sample_incept"+str(i)+".onnx"


    #     if ENABLE_PDQSEL_MODE:
    #         full_sec, under_mem, mac_count = run_test(test_model_path)
    #         if under_mem:
    #             pass_cons += 1
    #             total_MAC_gap += mac_count
    #     elif ENABLE_MICROGRAPH_MODE:
    #         full_sec, peak_after_sp, mac_count = run_test(test_model_path)
    #         if peak_after_sp <= AVAILABLE_VM:
    #             pass_cons += 1
    #             total_MAC_gap += mac_count

        
    #     total_time += full_sec

    #     if full_sec > max_time:
    #         max_time = full_sec
        
    # #if ENABLE_PDQSEL_MODE:
    # print(f"pass_cons: {pass_cons}, total_percent: {pass_cons/2}\n")
    # avg_time = total_time / 1

    # if pass_cons > 0:
    #     avg_MAC_gap = total_MAC_gap / pass_cons
    # else:
    #     avg_MAC_gap =0

    # print(f"Total_time: {total_time}\n")
    # print(f"avg_time: {avg_time}\n")
    # print(f"max_time: {max_time}\n")

    # print(f"avg_MAC_gap: {avg_MAC_gap}\n")
    # print(f"max_MAC_gap: {max_MAC_gap}\n")