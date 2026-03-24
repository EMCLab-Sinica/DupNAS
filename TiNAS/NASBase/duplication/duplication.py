import os, sys
import torch
import time
import copy
from os.path import dirname, realpath

import pprint
from pprint import PrettyPrinter
import onnx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
#import networkx as nx
#import pygraphviz as pgv
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import math

#sys.path.append("../..")
#sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from NASBase import utils as utils
from NASBase import file_utils as file_utils
from NASBase.model.common_utils import get_supernet
from settings import Settings, arg_parser, load_settings
from collections import defaultdict


AVAILABLE_VM = 128*1024

inplace_op=['Relu', 'Softmax','MaxPool', 'GlobalAveragePool', 'Squeeze', 'Add', 'Mul', 'Reshape',  'BatchNormalization', 'Sigmoid', 'Dropout','LRN', 'DequantizeLinear', 'QuantizeLinear', 'LeakyRelu','Split', 'Slice']
    



#----- function definition -----#
def add_node(node, node_list):
    node_list=node_list
    if node not in node_list:
        node_list.append(node)
        #print("Node",node, "is appended into node list.")
    #else:
        #print("Node",node, "alreadt exists.")
# search for nodes'dependency
    return node_list 

def add_adjacency(start,end,ad_list,node_list):
    ad_list=ad_list
    node_list=node_list
    tmp=[]
    if start not in node_list:
        n_list = add_node(start, node_list)
        node_list=n_list
        tmp.extend(end)
        ad_list[start]=tmp
        #print("Edge ",start," -> ",end, "is saved into ad_list.")
    else:
        tmp.extend(ad_list[start])
        tmp.extend(end)
        ad_list[start]=tmp

    return ad_list, node_list
        #print("Edge ",start," -> ",end, "is saved into ad_list.")
        
# find which nodes' input is same as the current output 
def find_node_by_inout(nodes, out):
    idx_list=[]
    for idx,element in enumerate(nodes):
        for input_ in element.input:
            if input_ == out:
                idx_list.append(idx)
    return idx_list

# save data info to data_usage[]
def get_shape_of_node(onnx_model,node_,output_,node_list,data_usage,d_cnt):
    data_usage=data_usage
    dd_cnt=d_cnt
    g = onnx_model.graph
    for idx in g.value_info:
        if idx.name == g.node[node_].output[output_]:
            new_data={}
            new_data['data'] = dd_cnt
            new_data['start'] = node_
            new_data['end'] = find_node_by_inout(g.node, g.node[node_].output[output_])
            if not new_data['end']:
                new_data['end'].append(node_)
            #print(new_data)
            kill=0
            #print("new_data['end']:", new_data['end'])
            for nend in new_data['end']:
                #print(node_list.index(nend))
                #print(max(kill, node_list.index(nend)))
                kill = max(kill, node_list.index(nend))
                #print(kill)
            new_data['kill'] = node_list[kill]
            shape_mul=1
            dimv=[]
            for dim_ in idx.type.tensor_type.shape.dim:
                dimv.append(dim_.dim_value)
                shape_mul*=dim_.dim_value
            new_data['dim'] = dimv
            new_data['size'] = shape_mul
            #print(new_data)
            data_usage.append(new_data)
            dd_cnt+=1
    return data_usage, dd_cnt


def find_node(name,node_list):
    for n, element in enumerate(node_list):
        if  element == name:
            return n

# allocate data to usable buffer and return the # of used buffer in current node
def buffer_alloc(node_status, data,inplace,buff,data_usage):
    data_usage=data_usage
    buff=buff
    if data not in buff:
        allo = 0
        if inplace:
            if node_status['type'] == 'Split':
                if node_status['input'][0] in buff:
                    loc = buff.index(node_status['input'][0])
                    buff[loc] = data
                    data_usage[data]['buffer'] = loc
                    allo = 1
                    inplace= False
                else:
                    allo = 0
                    inplace= False
            else:   
                if node_status['input']:
                    #print(buffer)
                    #print(node_status['input'][0])
                    loc = buff.index(node_status['input'][0])
                    buff[loc] = data
                    data_usage[data]['buffer'] = loc
                    allo = 1
                    inplace= False
                    
        if allo == 0:
            #print(buffer,node_status['input'][0])
            if None in buff:
                usable = buff.index(None)
                buff[usable] = data
                data_usage[data]['buffer'] = usable
            else :
                buff.append(data)
                data_usage[data]['buffer'] = len(buff)-1
    return buff, data_usage

# check if any buffer can be released
def check_buffer_release(nidx,buff,data_usage):
    buff=buff
    for idx, buf in enumerate(buff):
        if buf is not None:
            if data_usage[buf]['kill'] == nidx:
                buff[idx] = None
    return buff
    
def buffer_cnt(buff):
    if None in buff:
        indexes = [i for i, j in enumerate(buff) if j == None]
        return (len(buff)-len(indexes))
    else:
        return (len(buff))

def peak_mem_estimation(buff,data_usage):
    peak=0
    #print(buff)
    for buf in buff:
        if buf is not None:
            ds=data_usage[buf]['size']
            peak += ds
    return peak

def check_initializer(graph,input_name):
    for gin in graph.initializer:
        if gin.name == input_name:
            return True
    return False


def find_peak_node(mem_IOB, mem_constriant):
    #find peak nodes
    peak_node=[]
    #mem_constriant = 60*1024
    mem_constriant = mem_constriant
    print(mem_constriant)
    for midx,ms in enumerate(mem_IOB):
        mem_op = ms[1]+ms[2]+ms[3]
        if mem_op >= mem_constriant:
            peak_node.append(midx)

    return peak_node

def sort_peak_nodes_by_io_sum(peak_node, mem_IOB):
    # Sorting the peak_node list in-place based on the sum of input size and output size
    peak_node.sort(key=lambda pn: mem_IOB[pn][1] + mem_IOB[pn][2], reverse=True)
    return peak_node

def check_valid_peak_node_list(pn, dup_path):
    
    for dp in dup_path:
        if (pn >= int(dp['start'])) and (pn <= int(dp['end'])):
            return False  #already in path
    return True

def find_dup_path(peak_node, mem_IOB, Total_nodes, op_type):
    dup_path=[]
    

    for pn in peak_node:
        if len(dup_path)>0 and not check_valid_peak_node_list(pn, dup_path):
            continue

        cur_node = pn #middle node of path
        dup_check = True
        check_s = True
        s_node = cur_node
        #find start node of path

        mini_mem = mem_IOB[cur_node][1]
        existinp = 0
        for exdu in dup_path:
            if exdu['start']<=cur_node and cur_node<=exdu['end']:
                existinp = 1
                break

        if existinp == 0 and (mem_IOB[pn][3] != mem_IOB[pn-1][3]):    #no upper node in the same line
            s_node = cur_node
        else: #(mem_IOB[pn][3]==mem_IOB[pn-1][3]):
            for exdu in dup_path:
                if exdu['start']<=cur_node-1 and cur_node-1<=exdu['end']:
                    existinp = 1
                    break

            if existinp == 0 and (mem_IOB[pn][1]>=mem_IOB[pn-1][1]):
                s_node = cur_node-1
                mini_mem = mem_IOB[s_node][1]
                check_s = True
                print("s_node = ", s_node, "for peak ", pn)
                            #peak_node.pop(peak_node.index(s_node))
            else:
                for scn in range(cur_node-1,0,-1):
                    for exdu in dup_path:
                        if exdu['start']<=scn and scn<=exdu['end']:
                            existinp=1
                            break
                    if existinp == 0  and (mini_mem > mem_IOB[scn][1]) and (mem_IOB[cur_node][1] >= mem_IOB[scn][1]) and (mem_IOB[cur_node][3] == mem_IOB[scn][3]):
                        s_node = scn
                        mini_mem = mem_IOB[s_node][1]
                        check_s = True
                        print("s_node = ", s_node, "for peak ", pn)
                    else:
                        if (scn-1 >= 0) and (mini_mem > mem_IOB[scn-1][1]):
                            continue
                        else:
                            break
        
            mini_mem_in = mini_mem                    
                #if (check_s):          
        #find end node of path
        e_node = cur_node
        mini_mem = mem_IOB[e_node][2]
        if(cur_node+1 == Total_nodes-1) or (op_type[cur_node+1]=='Reshape') or (op_type[cur_node+1]=='Gemm'):
            dup_check = False
            continue

        existinp=0
        for exdu in dup_path:
            if exdu['start']<=cur_node+1 and cur_node+1<=exdu['end']:
                existinp=1
                break
        if existinp == 0 and (mem_IOB[cur_node][3]==mem_IOB[cur_node+1][3]) or (op_type[cur_node+1]=='Concat'):
            if (mem_IOB[cur_node][2]>=mem_IOB[cur_node+1][2]):
                e_node = cur_node+1
                mini_mem = mem_IOB[e_node][2]
                
            e_choices_num = Total_nodes-1-e_node
            for ecn in range(e_choices_num):
                if cur_node+2+ecn == Total_nodes-1:
                    break
                
                for exdu in dup_path:
                    if exdu['start']<=cur_node+2+ecn and cur_node+2+ecn<=exdu['end']:
                        existinp=1
                        break
                    if existinp == 0 and (mini_mem > mem_IOB[cur_node+2+ecn][2]) and (mem_IOB[cur_node][2] >= mem_IOB[cur_node+2+ecn][2]) and (op_type[cur_node+2+ecn]!='Reshape') and (op_type[cur_node+2+ecn]!='Gemm'):
                        if (mem_IOB[e_node][3] == mem_IOB[cur_node+2+ecn][3]) or (op_type[cur_node+2+ecn]=='Concat'):
                            e_node = cur_node+2+ecn
                            mini_mem = mem_IOB[e_node][2]
                    else:
                        if (cur_node+3+ecn < Total_nodes) and (mini_mem > mem_IOB[cur_node+3+ecn][2]) and (op_type[cur_node+3+ecn]!='Reshape') and (op_type[cur_node+3+ecn]!='Gemm'):
                            continue
                        else:
                            break
        else:
            dup_check = False
            print("Can not duplicate! no feasible end node")
    else:
        dup_check = False
        print("Can not duplicate! no feasible start node")

    if dup_check:
        new_path={}
        new_path['start'] = s_node
        new_path['peak'] = cur_node
        new_path['end'] = e_node
        dup_path.append(new_path)
        print("[start, peak, end] : ", s_node, cur_node,"(", op_type[cur_node], ") " , e_node,)
    else:
        print("Peak node ", cur_node,"(", op_type[cur_node], ") " ,"can not find duplication.")

    return peak_node, dup_path

# ---------- Topological Sort ----------#
#Python program to print topological sorting of a DAG

 
#Class to represent a graph
class Graph:
    def __init__(self,vertices):
        self.graph = defaultdict(list) #dictionary containing adjacency List
        self.V = vertices #No. of vertices
    
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # A recursive function used by topologicalSort
    def topologicalSortUtil(self,v,visited,stack):
 
        # Mark the current node as visited.
        visited[v] = True
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
 
        # Push current vertex to stack which stores result
        stack.insert(0,v)
        
    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False]*self.V
        stack =[]
        
        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in reversed(range(self.V)):  #for i in reversed(range(self.V)):
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
 
        # return contents of stack
        return stack




def model_tracing(onnx_path, onnx_name):


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

    pp = PrettyPrinter(width=150)
    onnx_model = onnx.load_model(onnx_path+onnx_name+'.onnx')
    model_fig_size = {'fig_d': (10,15), 'fig_m': (10,4), 'axis':10}
    PLOTDAG = True

    #----- load onnx -----#


#----- shape_inference -----# let onnx model can show all shapes (dims)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred_model,onnx_path+onnx_name+'_inferred.onnx')  # save onnx w/ shapes 
    model = inferred_model
    g=model.graph
    Total_nodes = len(g.node)

    nocnt = 0
    curnode = 0
    while nocnt < Total_nodes:
        if len(g.node[curnode].input) == 0:
            g.node.pop(curnode)
            nocnt += 1
        else:
            allinput_from_initializer = True
            for input_ in g.node[curnode].input:
                tmp = check_initializer(g,input_)
                allinput_from_initializer = allinput_from_initializer & tmp
            if allinput_from_initializer:
                g.node.pop(curnode)
                nocnt += 1  
            else:
                curnode += 1
                nocnt += 1
    
    #nocnt = 0
    #curnode = 0
    #while nocnt < Total_nodes:
    #   if len(g.node[curnode].input) == 0:
    #       g.node.pop(curnode)
    #       nocnt += 1
    Total_nodes = len(g.node)

    if Total_nodes==0:
        return 0, False, None, None, None

    for i in range(Total_nodes):
        op_type.append(g.node[i].op_type)
        start=i
        end=[]
        for out_ in g.node[i].output:
            if 'mask' in out_:
                g.node[i].output.pop()
            else:
                end.extend(find_node_by_inout(g.node, out_))
        #print('Node: ',start, ' -> Node ',end)
        ad_list,node_list = add_adjacency(start,end,ad_list,node_list)


    print('Total number of nodes: ',Total_nodes)
    #print('Node list: ', node_list)
    #print('Op types: ', op_type)


    
    #print(node_list[Total_nodes-1]+1)
#   res = []
#   tpg = Graph(node_list[Total_nodes-1]+1)
#   for n in ad_list:
#       for ad in ad_list[n]:
#           if len(ad_list[n]):
#               tpg.addEdge(n,ad)

#   res = tpg.topologicalSort()

#   for i, r in reversed(list(enumerate(res))):
#       if r not in node_list:
#           res.pop(i)

#   with open(onnx_path+onnx_name+'_TP_sort.txt', 'w') as f: 
#       f.write('Topological Sort of the given graph\n')
#       f.write(str(res))

#   op_type=[]
#   for r in res:
#       op_type.append(op_type[r])
#   node_list = res

# #node_list=[0, 1, 2, 3, 4, 5, 9, 6, 10, 7, 11, 8, 12, 13] #s3
# #node_list=[0, 1, 2, 3, 5, 6, 9, 10, 4, 7, 8, 11, 12, 13]  #s2
# #node_list=[0, 1, 2, 3, 5, 4, 6, 7]  #s1

#   for r in node_list:
#       op_type.append(op_type[r])
    
#   print ("Topological Sort of the given graph")
#   print (node_list)
#   print('Op types: ', op_type)

# ----- generate data usage ----- #
# model IFM
# print(g.node)
    new_data={}
    new_data['data'] = 0
    new_data['start'] = None
    new_data['end'] = find_node_by_inout(g.node, g.input[0].name)
    kill = 0
    for nend in new_data['end']:
        kill = max(kill, node_list.index(nend))
    new_data['kill']=node_list[kill]
    shape_mul = 1
    dimv=[]
    for dim_ in g.input[0].type.tensor_type.shape.dim:
        dimv.append(dim_.dim_value)
        shape_mul *= dim_.dim_value
    new_data['dim'] = dimv
    new_data['size'] = shape_mul

    data_usage.append(new_data)  # save first data (d0: model IFM) info to data_usage[]
#peak_list = [i for i, j in enumerate(peak_for_plt) if j == max(peak_for_plt)]

#other data         
    for i ,tmp1 in enumerate(g.node):
        if i in node_list:
            for j, tmp2 in enumerate(g.node[i].output):
                data_usage,d_cnt = get_shape_of_node(model,i,j,node_list,data_usage,d_cnt)

    if data_usage[len(data_usage)-1]['start'] != data_usage[len(data_usage)-1]['kill']:
        new_data = {}
        new_data['data'] = len(data_usage)
        new_data['start'] = data_usage[len(data_usage)-1]['kill']
        new_data['end'] = []
        new_data['end'].append(data_usage[len(data_usage)-1]['kill'])
        new_data['kill'] = data_usage[len(data_usage)-1]['kill']
        shape_mul = 1
        dimv=[]
        for dim_ in g.output[0].type.tensor_type.shape.dim:
            dimv.append(dim_.dim_value)
            shape_mul *= dim_.dim_value
        new_data['dim'] = dimv
        new_data['size'] = shape_mul
        data_usage.append(new_data) 

    #print('\nData_usage:')
    #for du in data_usage:
    #   print(du)
    #print("data_usage:")
    #print(data_usage)

# ----- record node_status ----- # = {name: op. idx, input:[data idx list], output:[data idx list], target_mem: peak_mem_size_plain, peak_mem:peak_mem_size_cur}
    for n in range(Total_nodes):
        new_status={}
        new_status['name'] = n
        new_status['input']=[]
        new_status['output']=[]
        new_status['target_mem']=0
        new_status['peak_mem']=0
        new_status['type']=op_type[n]
        if new_status['type']=="Conv":
            for a in g.node[n].attribute:
                if a.name == "kernel_shape":
                    new_status['kernel']=a.ints
                if a.name == "pads":
                    new_status['pads']=a.ints
                if a.name == "strides":
                    new_status['strides']=a.ints
        else:
            new_status['kernel'] = [0,0]
            new_status['pads'] = [0,0,0,0]
            new_status['strides'] = [0,0]
        if new_status['type'] in inplace_op:
            new_status['inplace'] = True
        else:
            new_status['inplace'] = False
        #if USEPATCH:
        #   new_status['patch'] = False
        new_status['bufnum'] = 0
        node_status.append(new_status)
      
    #pp.pprint(node_status)


    for dr in data_usage:
        if dr['start'] is not None:
            start_node=find_node(dr['start'],node_list)
            #start_node=dr['start']
            node_status[start_node]['output'].append(dr['data'])
            if dr['start'] != dr['kill']:
                for out_ in dr['end']:
                    end_node=find_node(out_,node_list)
                    #end_node=out_
                    node_status[end_node]['input'].append(dr['data'])
        else:
            for out_ in dr['end']:
                end_node=find_node(out_,node_list)
                #end_node=out_
                node_status[end_node]['input'].append(dr['data'])


# ----- buffer allocation ----- #
    used_buf_cnt=0
    peak_for_plt=[]
    for n in range(Total_nodes):
        mem_IOB.append([n,0,0,0])
        mem_stacked.append([n,0,0])

    for i,n in enumerate(node_list):
        for indata in node_status[n]['input']:
            buff, data_usage = buffer_alloc(node_status[n], indata, False, buff, data_usage)
            mem_IOB[i][1] += data_usage[indata]['size']
        if node_status[n]['inplace']:
            inplace = True
        else:
            inplace = False
    
        for outdata in node_status[n]['output']:
            buff, data_usage= buffer_alloc(node_status[n], outdata, inplace, buff, data_usage)
            if not inplace:
                mem_IOB[i][2]+=data_usage[outdata]['size']
        used_buf_cnt = buffer_cnt(buff)
        #print('Node ', node_status[n]['name'], ' Buffer usage: ', buffer , ' # of used buffer: ',used_buf_cnt)
        buf_num_per_op.append(used_buf_cnt)
        node_status[n]['bufnum'] = used_buf_cnt
        
        # peak memory
        #print(buff)

        peak_mem = peak_mem_estimation(buff,data_usage)
    #if op_type[i]=='Split':
    #   for idx,so in enumerate(node_status[i]['output']):
    #       if idx != 0:
    #           peak_mem+=data_usage[so]['size']
    #           mem_stacked[i][1]+=data_usage[so]['size']
                
        node_status[n]['peak_mem'] = peak_mem
        if peak_mem > (mem_IOB[i][1]+mem_IOB[i][2]):
            mem_IOB[i][3]=peak_mem - (mem_IOB[i][1]+mem_IOB[i][2])
        else:
            mem_IOB[i][3]=0

        peak_for_plt.append(peak_mem)

    # assume op is finishe, release buffer or not? (data killed?) 
    #print('Buffer usage: ', buffer)
        buff = check_buffer_release(n,buff,data_usage)
    
#print(mem_stacked) 
#pp.pprint(node_status)
    
    #print('\nNumber of buffers used per Op. :', buf_num_per_op)

    # targrt memory requirement 
    for n in range(Total_nodes):
        mem_stacked[n][1]=mem_IOB[i][1]+mem_IOB[i][2]
        mem_stacked[n][2]=mem_IOB[i][3]
        if(mem_IOB[n][2]==0):
            mem_IOB[n][2] = mem_IOB[n][1]

    target_mem_inplain = 0
    for n in node_list:
        maxin = 0
        maxout = 0
        # print(n['input'])
        for nin in node_status[n]['input']:
            maxin = max(maxin,data_usage[nin]['size'])
    
        for nout in node_status[n]['output']:
            maxout = max(maxout,data_usage[nout]['size'])
    
        if node_status[n]['inplace']:
            node_status[n]['target_mem'] = max(maxin, maxout)
        else:
            node_status[n]['target_mem'] = maxin+maxout
    
        target_mem_inplain = max(target_mem_inplain, node_status[n]['target_mem'])

#pp.pprint(node_status)

#----- save to txt file -----#
    peak_list = [node_list[i] for i, j in enumerate(peak_for_plt) if j == max(peak_for_plt)]
    target_list = [j for j in node_list if node_status[j]['target_mem'] == target_mem_inplain]

    with open(onnx_path+onnx_name+'_data_usage.txt', 'w') as f: 
        f.write('>>> Load onnx model \n')
        f.write('Nodes:' + str(node_list) + '\n')
        f.write('Type:' + str(op_type) + '\n')
        f.write('>>> TP Sort \n')
        f.write('Nodes:' + str(node_list) + '\n')
        f.write('Type:' + str(op_type) + '\n')
        f.write('\n')
        f.write('Peak memory: ' + str(max(peak_for_plt)) + ' in Node ' + str(peak_list) + '\n')
        f.write('Target memory: ' + str(target_mem_inplain) + ' in Node ' + str(target_list) + '\n')   
        f.write('\nNode_status:\n')
        for n in node_status:
            f.write(str(n)+'\n')
        f.write('\nData_usage:\n')
        for d in data_usage:
            f.write(str(d)+'\n')
        f.write('\nNumber of buffers used per Op. :'+ str(buf_num_per_op))
        f.write('\nMemory stacking per Op.: (Current, Others)\n')
        f.write(pprint.pformat(mem_IOB))
        f.write('\nAdjacency list: \n')
        f.write(pprint.pformat(ad_list))

    under_mem=False
##----- find peak node and path for duplication -----##
#
    peak_node=[]
    dup_path=[]
    mem_constriant = AVAILABLE_VM 
    peak_node = find_peak_node(mem_IOB,mem_constriant)
    if len(peak_node)>0:
        peak_node = sort_peak_nodes_by_io_sum(peak_node, mem_IOB)
        print("Peak nodes: ")   #print for check
        for pn in peak_node:
            print(pn,mem_IOB[pn][1], mem_IOB[pn][2], mem_IOB[pn][3])
        peak_node, dup_path = find_dup_path(peak_node, mem_IOB, Total_nodes, op_type)
        print(dup_path)
    else:
        under_mem=True
        print("All nodes under mem_constriant")
        

    
    #data={'cin':0, 'hin':0, 'win':0, 'cout':0, 'hout':0, 'wout':0,'k':0,'p':0,'s':0}
    perint=perout=0 
    
    q_choices=[2,4,8,16,32]
    can_dup=[]
    q_list_per_path=[]
    est_peak_per_path=[]
    est_lat_diff=0
    #catch dim info before dup 
    for didx, dp in enumerate(dup_path):
        before_dup=[]
        after_dup=[]
        dup_info=[]
        est_lat=[]
        est_mem=[]
        per_mem=[]
        buf_start=[]
        buf_end=[]
        del_lat_mem=[]
        perdup={}
        perdup['path']=didx
        total_lat=0
        peam_mem=0
        need_buf=0
        for dn in range(dp['start'],dp['end']+1,1):
            perno={}
            perno['cin']=data_usage[node_status[dn]['input'][0]]['dim'][1]
            perno['hin']=data_usage[node_status[dn]['input'][0]]['dim'][2]
            perno['win']=data_usage[node_status[dn]['input'][0]]['dim'][3]
            perno['cout']=data_usage[node_status[dn]['output'][0]]['dim'][1]
            perno['hout']=data_usage[node_status[dn]['output'][0]]['dim'][2]
            perno['wout']=data_usage[node_status[dn]['output'][0]]['dim'][3]
            perno['k']=node_status[dn]['kernel'][0]
            perno['p']=node_status[dn]['pads'][0]
            perno['s']=node_status[dn]['strides'][0]
            #calculate lat and mem before dup
            perno['eva_comp']=perno['k']*perno['k']*perno['cin']*perno['cout']*perno['hout']*perno['wout']
            perno['eva_access']=perno['hin']*perno['win']*perno['cin'] + perno['k']*perno['k']*perno['cin']*perno['cout']
            perno['eva_mem']=perno['hin']*perno['win']*perno['cin'] + perno['hout']*perno['wout']*perno['cout']
            perno['eva_buf']=mem_IOB[dn][3]
            total_lat+=perno['eva_comp']+perno['eva_access']
            if dn == dp['peak']:
                peak_mem = perno['eva_mem']
                need_buf = perno['eva_buf']
            before_dup.append(perno)
        
        print(before_dup)
        est_lat.append(total_lat)
        est_mem.append(peak_mem)
        per_mem.append(peak_mem)
        usable_q=[]
        usable_q.append(1)
        buf_start.append(0)
        buf_end.append(0)
        sel_q=0
        min_lat_per_mem=0
        q_comp=q_access=q_mem=0 

        for q in q_choices:
            if((q<before_dup[0]['hin']) and (q<before_dup[-1]['hin'])):
                
                #peak mem
                pidx=dp['start']-dp['peak']
                if before_dup[pidx]['k'] > before_dup[pidx]['s']:
                    kminuss=before_dup[pidx]['k']-before_dup[pidx]['s']
                else:
                    kminuss=0
                if before_dup[pidx]['s']==0:
                    mem=before_dup[pidx]['cin']* math.ceil((before_dup[pidx]['hin']+2*before_dup[pidx]['p'])/q)*(before_dup[pidx]['win']+2*before_dup[pidx]['p'])*q + before_dup[pidx]['cout']*(before_dup[pidx]['hin']+2*before_dup[pidx]['p']-before_dup[pidx]['k']+1) * math.ceil((before_dup[pidx]['win']+2*before_dup[pidx]['p']-before_dup[pidx]['k']+1)/q) + q*kminuss*(before_dup[pidx]['win']+2*before_dup[pidx]['p'])
                else:
                    mem=before_dup[pidx]['cin']* math.ceil((before_dup[pidx]['hin']+2*before_dup[pidx]['p'])/q)*(before_dup[pidx]['win']+2*before_dup[pidx]['p'])*q + before_dup[pidx]['cout']*math.ceil(math.ceil((before_dup[pidx]['hin']+2*before_dup[pidx]['p']-before_dup[pidx]['k']+1)/before_dup[pidx]['s']) * math.ceil((before_dup[pidx]['win']+2*before_dup[pidx]['p']-before_dup[pidx]['k']+1)/before_dup[pidx]['s'])/q) + q*kminuss*(before_dup[pidx]['win']+2*before_dup[pidx]['p'])
                if(mem<=est_mem[0]):
                    usable_q.append(q)
                    per_mem.append(mem)
                    
                    #buffer
                    if before_dup[0]['k'] > before_dup[0]['s']:
                        kminuss=before_dup[0]['k']-before_dup[0]['s']
                    else:
                        kminuss=0
                    if before_dup[0]['s']==0:
                        qmem=before_dup[0]['cin']* math.ceil((before_dup[0]['hin']+2*before_dup[0]['p'])/q)*(before_dup[0]['win']+2*before_dup[0]['p'])*q + before_dup[0]['cout']*(before_dup[0]['hin']+2*before_dup[0]['p']-before_dup[0]['k']+1) * math.ceil((before_dup[0]['win']+2*before_dup[0]['p']-before_dup[0]['k']+1)/q) + q*kminuss*(before_dup[0]['win']+2*before_dup[0]['p'])
                    else:
                        qmem=before_dup[0]['cin']* math.ceil((before_dup[0]['hin']+2*before_dup[0]['p'])/q)*(before_dup[0]['win']+2*before_dup[0]['p'])*q + before_dup[0]['cout']*math.ceil(math.ceil((before_dup[0]['hin']+2*before_dup[0]['p']-before_dup[0]['k']+1)/before_dup[0]['s']) * math.ceil((before_dup[0]['win']+2*before_dup[0]['p']-before_dup[0]['k']+1)/before_dup[0]['s'])/q) + q*kminuss*(before_dup[0]['win']+2*before_dup[0]['p'])
                    bufs=qmem*(q-1)

                    if before_dup[-1]['k'] > before_dup[-1]['s']:
                        kminuss=before_dup[-1]['k']-before_dup[-1]['s']
                    else:
                        kminuss=0
                    if before_dup[-1]['s']==0:
                        qqmem=before_dup[-1]['cin']* math.ceil((before_dup[-1]['hin']+2*before_dup[-1]['p'])/q)*(before_dup[-1]['win']+2*before_dup[-1]['p'])*q + before_dup[-1]['cout']*(before_dup[-1]['hin']+2*before_dup[-1]['p']-before_dup[-1]['k']+1) * math.ceil((before_dup[-1]['win']+2*before_dup[-1]['p']-before_dup[-1]['k']+1)/q) + q*kminuss*(before_dup[-1]['win']+2*before_dup[-1]['p'])
                    else:
                        qqmem=before_dup[-1]['cin']* math.ceil((before_dup[-1]['hin']+2*before_dup[-1]['p'])/q)*(before_dup[-1]['win']+2*before_dup[-1]['p'])*q + before_dup[-1]['cout']*math.ceil(math.ceil((before_dup[-1]['hin']+2*before_dup[-1]['p']-before_dup[-1]['k']+1)/before_dup[-1]['s']) * math.ceil((before_dup[-1]['win']+2*before_dup[-1]['p']-before_dup[-1]['k']+1)/before_dup[-1]['s'])/q) + q*kminuss*(before_dup[-1]['win']+2*before_dup[-1]['p'])
                    bufe=qqmem*(q-1)
                    
                    buf_start.append(bufs)
                    buf_end.append(bufe)
                    est_mem.append(mem+bufs+bufe)

                    # total latency
                    q_total_lat=0
                    for bd in before_dup:
                        if bd['k'] > bd['s']:
                            kminuss=bd['k']-bd['s']
                        else:
                            kminuss=0
                        if bd['s']==0:
                            comp=q*bd['k']*bd['k']*bd['cin']*bd['cout']*(math.ceil((bd['hin']+2*bd['p'])/q) + 1-bd['k']+kminuss) * (bd['win']+2*bd['p']+1-bd['k'])
                            access=bd['hin']*bd['win']*bd['cin'] + q*bd['k']*bd['k']*bd['cin']*bd['cout']
                        else:
                            comp=q*bd['k']*bd['k']*bd['cin']*bd['cout']*math.ceil((math.ceil((bd['hin']+2*bd['p'])/q) + 1-bd['k']+kminuss)/bd['s']) * math.ceil((bd['win']+2*bd['p']+1-bd['k'])/bd['s'])
                            access=bd['hin']*bd['win']*bd['cin'] + q*bd['k']*bd['k']*bd['cin']*bd['cout']
                        q_total_lat+=comp+access
                    est_lat.append(q_total_lat)
                    del_lat_mem.append((q_total_lat-est_lat[0])/(est_mem[0]-mem))
                else:
                    print("Memory is not reduced by q:",q)
                    continue

        print(usable_q)
        print(est_lat)
        print(est_mem)
        print(buf_start)
        print(buf_end)
        print(per_mem)
        print(del_lat_mem)

        min_mem_idx=0
        # minimize peak mem
        
        #print(est_mem)
        #print(est_lat)

        min_mem_idx=est_mem.index(min(est_mem))
        if min_mem_idx ==0:
            print("no dup:", est_mem[0], est_lat[0])
            est_peak_per_path.append(est_mem[0])
            q_list_per_path.append(1)
            can_dup.append(False)
        else:
            if est_mem[min_mem_idx]+need_buf < mem_constriant:
                print("Minimun memory after dup is under constraint:")
                print("Chosse: ", usable_q[min_mem_idx], "est_mem", est_mem[min_mem_idx],"need_buf", need_buf, "est_lat", est_lat[min_mem_idx])
                est_lat_diff+=(est_lat[min_mem_idx]-est_lat[0])
                est_peak_per_path.append(est_mem[min_mem_idx])
                q_list_per_path.append(usable_q[min_mem_idx])
                can_dup.append(True)
            else:
                print("Minimun memory after dup is not under constraint:")
                print("Minimun: ", usable_q[min_mem_idx],est_mem[min_mem_idx],est_lat[min_mem_idx])
                est_peak_per_path.append(est_mem[min_mem_idx])
                q_list_per_path.append(usable_q[min_mem_idx])
                can_dup.append(False)
            #print(len(usable_q),len(est_mem),len(est_lat), min_mem_idx)
            #print(usable_q)
        

        #print("If select del_lat/del_mem memory:")
        #balance_idx=del_lat_mem.index(min(del_lat_mem))
        #print(usable_q[balance_idx+1],est_mem[balance_idx+1],est_lat[balance_idx+1])

        perdup['before']=before_dup
        dup_info.append(perdup)

        fig, ax1 = plt.subplots()
        plt.title('Estimation memory and latency by differernt duplications')
        plt.xlabel('Duplication choices')
        ax2 = ax1.twinx()

        ax1.set_ylabel('Peak memory', color='tab:blue')
        ax1.bar(usable_q, est_mem, color='tab:blue',width=0.5)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2.set_ylabel('Path latency', color='black')
        ax2.plot(usable_q,est_lat,color='black', alpha=0.75)
        #ax2.plot([ 250 * (i + 1) for i in range(len(data_avg))], data_avg, color='black', alpha=1)
        ax2.tick_params(axis='y', labelcolor='black')

        fig.tight_layout()
        plt.savefig(onnx_path+onnx_name+'_dup_'+str(didx)+'.png', bbox_inches='tight') # save to png
        plt.show()

        fig, ax = plt.subplots(figsize=model_fig_size['fig_m'])
        width = 0.5
        ax.bar(usable_q, est_mem, width, label='Peak memory', color='#5881b8')
        plt.plot(usable_q, est_lat, 'b', label='Path latency')
        plt.xlabel('q choices')
        plt.title('Estimation memory and latency by differernt duplications ')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(onnx_path+onnx_name+'_dup_'+str(didx)+'.png', bbox_inches='tight') # save to png
        plt.show()
        plt.close()
        

#compu = q*k[i]*k[i]*c[i]*math.ceil((math.ceil((h[i]+2*p[i])/q) + 1-k[i]+kminuss)/s[i]) * math.ceil((w[i]+2*p[i]+1-k[i])/s[i])*c[i+1]
#access = h[i]*w[i]*c[i] + q*k[i]*k[i]*c[i]*c[i+1]
#mem= c[i]* math.ceil((h[i]+2*p[i])/q)*(w[i]+2*p[i])*q + c[i+1]*math.ceil(math.ceil((ℎ[i]+2*𝑝[i]-𝑘[i]+1)/𝑠[i]) * math.ceil((𝑤[i]+2*𝑝[i]-𝑘[i]+1)/𝑠[i])/q) + q*kminuss*(w[i]+2*p[i])

            #node_status[dn]['input'][0]
            #node_status[dn]['output'][0]
            #print(dn)
        #calculate lat and mem after dup
        



        
    #calculate lat and mem before dup
    #

#node_status[]['input'] node_status[]['onput']


        

#----- matplotlib.pyplot -----#
# ----- memory stacked figure -----#
    fig, ax = plt.subplots(figsize=model_fig_size['fig_m'])

    col=['Idx', 'Current', 'Others']
    df = pd.DataFrame(mem_stacked,columns=col)
# view data
# plot data in stack manner of bar type
#color_sns=sns.color_palette('Set2')
#ax = df.plot(x='Operations', kind='bar', stacked=True, color=color_sns, title='Memory requirement', width=0.2, figsize = model_fig_size['fig_m'], fontsize='10')
#plt.axhline(y=max(peak_for_plt), color='pink', linewidth=2, linestyle='-', label='Peak')
    width = 0.5
    ax.bar(df['Idx'], df['Current']/1024, width, label='Per-layer', color='#5881b8')

    ax.bar(df['Idx'], df['Others']/1024, width, bottom=df['Current']/1024, label='Per-layer (Residual)', color='#f7b64d')
#plt.axhline(y=target_mem_inplain, color='tab:orange', linewidth=2, linestyle='--', label='Target')
    yupper = max(peak_for_plt)*1.1/1024
    plt.ylim(0, yupper)
    plt.xlim(-1, Total_nodes)
    ax.set_xlabel('Block Index')
    ax.set_xticks(range(0,Total_nodes,10))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    ax.add_patch(
     patches.Rectangle(
        (-1, 256),
        Total_nodes+1,
        64,
        facecolor='#EA433566'
     ) ) 

    plt.savefig(onnx_path+onnx_name+'_memory_stacked.png', bbox_inches='tight') # save to png
    plt.show()
    plt.close()


#----- Data lifetime distribution -----#
# figure format setting
    plt.figure(figsize = model_fig_size['fig_d'])
#plt.title("Data lifetime distribution")
    ax = plt.axes()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    plt.rcParams['xtick.direction'] = 'in'
    ax.set_xticks(range(0,len(data_usage),model_fig_size['axis']))

# plot data lifetime distribution
    data_labels=[]
    color=sns.color_palette('Set2')
    line=['-',':']
    for dr in data_usage:
        #c_idx = dr['buffer']%len(color)
        if dr['start'] is None:
            c_idx = node_status[dr['kill']]['inplace']
        else:
            c_idx = node_status[dr['kill']]['inplace']
        
        if dr['start'] is not None:
            #print(dr['start'],max(dr['end']))
            data_labels.append(dr['data'])
            plt.plot([dr['data']-0.25,dr['data']+0.25], [node_list.index(dr['start']),node_list.index(dr['start'])], color=color[0], linestyle=line[0])
            plt.vlines(x = dr['data'], ymin = node_list.index(dr['start']), ymax = node_list.index(dr['start'])+1, linewidth = 3, colors=color[0], linestyle=line[0])
            for dd in range(node_list.index(dr['start'])+1, node_list.index(dr['kill'])):
                plt.vlines(x = dr['data'], ymin = dd, ymax = dd+1, linewidth = 3, colors=color[0])
            plt.vlines(x = dr['data'], ymin = node_list.index(dr['kill']), ymax = node_list.index(dr['kill'])+1, linewidth = 3, colors=color[0], linestyle=line[c_idx])
            plt.plot([dr['data']-0.25,dr['data']+0.25], [node_list.index(dr['kill'])+1,node_list.index(dr['kill'])+1], color=color[0], linestyle=line[0])
        else:
            plt.plot([-0.25,0.25], [0,0], color=color[0], linestyle=line[c_idx])
            plt.vlines(x = dr['data'], ymin = 0, ymax = 1, linewidth = 3, colors=color[0], linestyle=line[0])
            for dd in range(1, node_list.index(dr['kill'])):
                plt.vlines(x = dr['data'], ymin = dd, ymax = dd+1, linewidth = 3, colors=color[0], linestyle=line[0])
            plt.vlines(x = dr['data'], ymin = node_list.index(dr['kill']), ymax = node_list.index(dr['kill'])+1, linewidth = 3, colors=color[0], linestyle=line[c_idx])
            plt.plot([-0.25,0.25], [node_list.index(dr['kill'])+1,node_list.index(dr['kill'])+1], color=color[0], linestyle=line[0])
    plt.plot([0,0],[0,0],label='Non-inplace', color=color[0], linestyle=line[0])
    plt.plot([0,0],[0,0],label='Inplace', color=color[0], linestyle=line[1])

    plt.ylabel('Block Index')
    plt.xlabel('Data')
    y=np.arange(Total_nodes+1)
    yminor=np.arange(Total_nodes)+0.5
    ylabel= range(Total_nodes)
#ylabel=[op_type[i] for i in range(Total_nodes)]
#print(ylabel)
    plt.ylim(Total_nodes, 0)
#plt.xlim(0, len(data_usage))
    ax.set_yticks(range(0,Total_nodes,5))
#ax.set_yticklabels('')
#ax.set_yticks(yminor, minor=True)
#ax.set_yticklabels(ylabel, minor=True, fontsize='10')
    ax.tick_params(axis= 'y', which='minor', left=False, right=False)
    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(onnx_path+onnx_name+'_data_lifetime.png', bbox_inches='tight') # save to png
    plt.show()
    plt.close()





##---latency: move to upper func; need subnet obj info---##

    cin=[]
    hin=[]
    win=[]
    cout=[]
    hout=[]
    wout=[]
    k=[]
    s=[]
    for i in range(Total_nodes):
        cin.append(0)
        hin.append(0)
        win.append(0)
        cout.append(0)
        hout.append(0)
        wout.append(0)
        k.append(0)
        s.append(1) 
    #oru 
#k=[1,2,2]
#s=[1,2,4]
#s1
#k=[1,1,1,2,2,2,2,1]
#s=[1,1,1,2,2,4,4,1]
#s2 
#k=[1,1,1,2,2,1,1,1,1 ,2,2,2,2,1]
#s=[1,1,1,2,2,1,1,1,1,4,4,4,4,1]
#s3
#k=[1,1,1,1,1,2,2,2,2,2,2,2,2,1]
#s=[1,1,1,1,1,2,2,2,2,4,4,4,4,1]
    
#input
    cin[0]=g.input[0].type.tensor_type.shape.dim[1].dim_value
    hin[0]=g.input[0].type.tensor_type.shape.dim[2].dim_value
    win[0]=g.input[0].type.tensor_type.shape.dim[3].dim_value


    for n in range(Total_nodes):
        for idx in g.value_info:
            if idx.name == g.node[n].input[0]:
                cin[n]=idx.type.tensor_type.shape.dim[1].dim_value
                if len(idx.type.tensor_type.shape.dim)>2:
                    hin[n]=idx.type.tensor_type.shape.dim[2].dim_value 
                if len(idx.type.tensor_type.shape.dim)>3:
                    win[n]=idx.type.tensor_type.shape.dim[3].dim_value 
            if idx.name == g.node[n].output[0]:
                cout[n]=idx.type.tensor_type.shape.dim[1].dim_value
                if len(idx.type.tensor_type.shape.dim)>2:
                    hout[n]=idx.type.tensor_type.shape.dim[2].dim_value 
                if len(idx.type.tensor_type.shape.dim)>3:
                    wout[n]=idx.type.tensor_type.shape.dim[3].dim_value 
        
        if node_status[n]['type'] == 'Conv':
            𝑘[n]=node_status[n]['kernel'][0]
            #p[n]=node_status[n]['pads'][0]
            s[n]= 1 #node_status[n]['strides'][0]

    cout[Total_nodes-1]=g.output[0].type.tensor_type.shape.dim[1].dim_value
    if len(idx.type.tensor_type.shape.dim)>2:
        hout[Total_nodes-1]=g.output[0].type.tensor_type.shape.dim[2].dim_value
    if len(idx.type.tensor_type.shape.dim)>3:
        wout[Total_nodes-1]=g.output[0].type.tensor_type.shape.dim[3].dim_value 

    #f.write('Latency per Op. : \n')
    #for s,i in enumerate(node_list):
    #   f.write(node_status[i]['type'] +'\t'+ str(latency[i]) + '\n')
    #f.write('Total latency :' + str(total_lan) + ' \n')

    latency=[] 
    for i,n in enumerate(node_list):
        if node_status[n]['type'] == 'Conv':
            #print(k[n],cin[n],hin[n],win[n],cout[n],math.ceil((hin[n]+2*1-𝑘[n]+1)/s[n]),math.ceil((win[n]+2*1-𝑘[n]+1)/s[n]),hout[n], wout[n])
            tmp=𝑘[n]*𝑘[n]*cin[n]*math.ceil((hin[n]+2*1-𝑘[n]+1)/s[n]) * math.ceil((win[n]+2*1-𝑘[n]+1)/𝑠[n])*cout[n] + k[n]*k[n]*cin[n]*cout[n]*6
        else:
            tmp= 0
        if n==0:
            tmp+= (hin[n]*win[n]*cin[n])*6
        latency.append(tmp)
    
    print(latency)

    total_lan=0
    for i in latency:
        total_lan+=i
    
    #print("total latency:" total_lan)
    

##-----------------------------------------##
#
    if under_mem: #no need to dup
        print("total_lan = ", total_lan, " no need to dup") 
        return total_lan, under_mem, None, None, None

    else:
        print("total_lan = ", total_lan, " lan_diff = ", est_lat_diff) 

        if False in can_dup:
            under_mem = False
        else:
            total_lan+=est_lat_diff
            under_mem = True

        return total_lan, under_mem, dup_path, est_peak_per_path, q_list_per_path
#print('\nAdjacency list: ')
#pp.pprint(ad_list)
#pp.pprint(onnx_model.graph.node)






#!/usr/bin/env python
# coding: utf-8






