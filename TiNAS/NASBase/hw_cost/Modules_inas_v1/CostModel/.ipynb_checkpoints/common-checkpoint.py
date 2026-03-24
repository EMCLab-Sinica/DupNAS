import sys, os
from pprint import pprint
import numpy as np
from time import perf_counter 
import inspect

import pandas as pd


from ....model.common_types import OPTYPES



#######################
# Dim calcs
#######################
# def _calc_ofm_dim(i, k, s=1, p=0):
#     o = int(((i-k + (2*p))/s) + 1)      
#     return o

def _calc_conv_ifm_tile_size(Tr, Tc, Kh, Kw, stride = 1, padding=0, layer_type='CONV'):    
    if (layer_type == 'CONV'):
        #print ("------------- CONV -------------")
        Tri = ((stride*Tr) + Kh - stride)
        Tci = ((stride*Tc) + Kw - stride)
        #print ("------------- CONV -------------")
    elif (layer_type == 'FC'):
        #print ("------------- FC -------------")
        Tri = ((stride*Tr) + Kh - stride)
        Tci = ((stride*Tc) + Kw - stride)
        #print ("------------- FC -------------")
    elif (layer_type == 'POOL'):
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unsupported yet")
    elif (layer_type == "GAVGPOOL") or (layer_type == "BN") or (layer_type == "RELU") or (layer_type == "ADD"):
        Tri = Tr
        Tci = Tc        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")

    return Tri, Tci


#######################
# Layers
#######################
def _get_layer_props(layer):
    if isinstance(layer['IFM'], dict):  # not of type Mat
        H = int(layer['IFM']['H']) if layer['IFM']['H'] is not None else None
        W = int(layer['IFM']['W']) if layer['IFM']['W'] is not None else None
        R = int(layer['OFM']['H']) if layer['OFM']['H'] is not None else None
        C = int(layer['OFM']['W']) if layer['OFM']['W'] is not None else None
        M = int(layer['OFM']['CH']) if layer['OFM']['CH'] is not None else None
        N = int(layer['IFM']['CH']) if layer['IFM']['H'] is not None else None
        Kw = int(layer['K']['W']) if layer['K']['W'] is not None else None
        Kh = int(layer['K']['H']) if layer['K']['H'] is not None else None
        stride = layer['stride']
    else:        
        H = layer['IFM'].h
        W = layer['IFM'].w
        R = layer['OFM'].h
        C = layer['OFM'].w
        M = layer['OFM'].ch
        N = layer['IFM'].ch
        Kw = layer['K'].w;  Kh = layer['K'].h
        stride = layer['stride']
    #Ri = (stride*R) + Kh - stride
    #Ci = (stride*C) + Kw - stride

    return (H, W, R, C, M, N, Kh, Kw, stride)

def check_fc(layer):
    if layer['type'] == "FC":
        if ("END" not in layer['name']): #FC_x
            if  (layer['stride'] == 1) and \
                (layer['K'].h == layer['IFM'].h) and (layer['K'].w == layer['IFM'].w) and \
                (layer['K'].ch == layer['IFM'].ch) and \
                (layer['K'].n == layer['OFM'].ch) and \
                (layer['OFM'].h == 1) and (layer['OFM'].w == 1):
                    return True
            else:
                    pprint(layer); return False        

        else:   #FC_END
            if (layer['IFM'].h == layer['IFM'].w) and (layer['OFM'].h == layer['OFM'].w) and \
            (layer['K'].h == layer['K'].w) and \
            (layer['K'].h == layer['IFM'].h) and (layer['K'].w == layer['IFM'].w) and \
            (layer['K'].ch == layer['IFM'].ch) and \
            (layer['K'].n == layer['OFM'].ch) and \
            (layer['OFM'].h == 1) and (layer['OFM'].w == 1):
                #(layer['stride'] == 1) and \
                return True
            else:
                pprint(layer); return False
    else:
        pprint(layer); return False

# check validity of conv layer properties
def check_conv(layer):
    if layer['type'] == "CONV":
        
        # -- 1D CONV
        if (layer['IFM'].w == 1 and layer['OFM'].w == 1 and layer['K'].w == 1): 
            if (layer['stride'] in [1,2]) and \
                (layer['IFM'].h > layer['K'].h) and \
                (layer['IFM'].h >= layer['OFM'].h) and \
                (layer['K'].ch == layer['IFM'].ch) or (layer['K'].ch == 1) and \
                (layer['K'].n == layer['OFM'].ch):
                    return True
            else:
                    pprint(layer); return False
        
        # -- 2D CONV (dw conv / pw conv / std conv)
        else:        
            if (layer['IFM'].h == layer['IFM'].w) and (layer['OFM'].h == layer['OFM'].w) and \
                (layer['K'].h == layer['K'].w) and \
                (layer['stride'] in [1,2]) and \
                (layer['IFM'].h >= layer['OFM'].h) and \
                ((layer['K'].ch == layer['IFM'].ch) or (layer['K'].ch == 1)) and \
                (layer['K'].n == layer['OFM'].ch):
                    
                    if (layer["pad"]>0):
                        return True
                    
                    elif (layer["pad"] == 0):
                        if (layer['IFM'].h >= layer['K'].h):          
                            return True
                        else:
                            pprint(layer); return False
                    else:                        
                        return False            
            else:
                    pprint(layer); return False            
    else:
        pprint(layer); return False


# check validity of pool layer properties
def check_pool(layer):
    if layer['type'] == "POOL":
        if (layer['IFM'].h == layer['IFM'].w) and (layer['OFM'].h == layer['OFM'].w) and \
           (layer['K'].h == layer['K'].w) and \
           (layer['stride'] == layer['K'].h) and \
           (layer['IFM'].h > layer['K'].h) and \
           (layer['IFM'].h > layer['OFM'].h) and \
           (layer['K'].ch == 1) and \
           (layer['K'].n == layer['OFM'].ch) and \
           (layer['IFM'].ch == layer['OFM'].ch):
                return True
        else:
                pprint(layer); return False
    
    elif layer['type'] == "GAVGPOOL":
        
        # -- 1D GAVGPOOL
        if (layer['IFM'].w == 1 and layer['OFM'].w == 1): 
            if  (layer['OFM'].h == 1) and (layer['OFM'].w == 1) and \
                (layer['IFM'].h > layer['OFM'].h) and \
                (layer['K'].h == layer['K'].w) and \
                (layer['IFM'].ch == layer['OFM'].ch):
                        return True
            else:
                        pprint(layer); return False
        
        
        # -- 2D GAVGPOOL
        else:
            if (layer['IFM'].h == layer['IFM'].w) and \
               (layer['OFM'].h == 1) and (layer['OFM'].w == 1) and \
               (layer['IFM'].h > layer['OFM'].h) and \
               (layer['K'].h == layer['K'].w) and \
               (layer['IFM'].ch == layer['OFM'].ch):
                    # (layer['stride'] == 1) and \
                    # (layer['IFM'].h == layer['K'].h) and \
                    # (layer['K'].ch == layer['IFM'].ch) and \
                    # (layer['K'].n == layer['OFM'].ch) and \
                        return True
            else:
                        pprint(layer); return False    
    else:
        pprint(layer); return False



# check validity of BN layer properties
def check_bn(layer):
    if layer['type'] == "BN":
        if  (layer['stride'] == None) and \
            (layer['K'].h == None) and (layer['K'].w == None) and (layer['K'].ch == None) and (layer['K'].n == None) and \
            (layer['IFM'].h == layer['OFM'].h) and (layer['IFM'].w == layer['OFM'].w) and \
            (layer['IFM'].ch == layer['OFM'].ch):
                    return True
        else:
                    pprint(layer); return False
    else:
        pprint(layer); return False
    
    
def check_relu(layer):
    if layer['type'] == "RELU":
        if  (layer['stride'] == None) and \
            (layer['K'].h == None) and (layer['K'].w == None) and (layer['K'].ch == None) and (layer['K'].n == None) and \
            (layer['IFM'].h == layer['OFM'].h) and (layer['IFM'].w == layer['OFM'].w) and \
            (layer['IFM'].ch == layer['OFM'].ch):
                    return True
        else:
                    pprint(layer); return False
    else:
        pprint(layer); return False
    

def check_add(layer):
    if layer['type'] == "ADD":
        if  (layer['stride'] == None) and \
            (layer['K'].h == None) and (layer['K'].w == None) and (layer['K'].ch == None) and (layer['K'].n == None) and \
            (layer['IFM'].h == layer['OFM'].h) and (layer['IFM'].w == layer['OFM'].w) and \
            (layer['IFM'].ch == layer['OFM'].ch):
                    return True
        else:
                    pprint(layer); return False
    else:
        pprint(layer); return False
    
    

#######################
# Power cycles
#######################


def _num_tiles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, layer_type='CONV', op_type=OPTYPES.O_CONV2D):
    if (layer_type == 'CONV') or (layer_type == 'FC'):
        if op_type == OPTYPES.O_CONV2D_DW or op_type == OPTYPES.O_CONV1D_DW:
            # Not using N/Tn as Tn=1 for DWCONV
            nt = np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm)

        else:
            nt = np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm) * np.ceil(N/Tn)
        
    elif (layer_type == 'POOL'):
        nt = np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm)
    elif (layer_type == "GAVGPOOL"):
        nt = np.ceil(H/Tr) * np.ceil(W/Tc) * np.ceil(M/Tm)
    elif (layer_type == "BN") or (layer_type == "RELU") or (layer_type == "ADD"):
        nt = np.ceil(H/Tr) * np.ceil(W/Tc) * np.ceil(M/Tm)
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")        
    return nt

# num power cycles where n=0 and n>0
def _num_pow_cycles(H, W, R, C, M, N, Tr, Tc, Tm, Tn, S, inter_lo, layer_type='CONV', op_type=OPTYPES.O_CONV2D):
    
    if (layer_type == 'CONV') or (layer_type == 'FC'):
        if op_type == OPTYPES.O_CONV2D_DW or op_type == OPTYPES.O_CONV1D_DW:
            npc = (np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm))/S
            # Not using N/Tn as Tn=1 for DWCONV
            # how many power cycles where n=0 ?
            if inter_lo == 'reuse_I' or inter_lo == 'reuse_W':
                n0 = (np.ceil(R/Tr) * np.ceil(C/Tc)) / S        
            elif inter_lo == 'reuse_O':
                n0 = (np.ceil(R/Tr) * np.ceil(C/Tc))
            else:
                sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")

        else:
            npc = (np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm) * np.ceil(N/Tn))/S
            # how many power cycles where n=0 ?
            if inter_lo == 'reuse_I' or inter_lo == 'reuse_W':
                n0 = (np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm)) / S        
            elif inter_lo == 'reuse_O':
                n0 = (np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm))
            else:
                sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")

        ngt0 = npc - n0

    elif (layer_type == 'POOL'):
        npc = (np.ceil(R/Tr) * np.ceil(C/Tc) * np.ceil(M/Tm))/S
        n0 = npc
        ngt0 = npc - n0
    
    elif (layer_type == "GAVGPOOL"):
        npc = (np.ceil(H/Tr) * np.ceil(W/Tc) * np.ceil(M/Tm))/S
        n0 = npc
        ngt0 = npc - n0        
                
    elif (layer_type == "BN") or (layer_type == "RELU") or (layer_type == "ADD"):
        npc = (np.ceil(H/Tr) * np.ceil(W/Tc) * np.ceil(M/Tm))/S
        n0 = npc
        ngt0 = npc - n0        
        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")

    return npc, n0, ngt0



#######################
# Buffers
#######################


def _vm_buff_size(Kh, Kw, Tri, Tci, Tr, Tc, Tm, Tn, 
                  inter_lo, S, layer_type='CONV', op_type=OPTYPES.O_CONV2D):
    
    if (layer_type == 'CONV') or (layer_type == 'FC'):
        if op_type == OPTYPES.O_CONV2D_DW or op_type == OPTYPES.O_CONV1D_DW:
            # Depthwise conv, where Tn=1
            if inter_lo == 'reuse_I' or inter_lo == 'reuse_W':
                B_in = (Tri * Tci * Tm)
                B_w = (Kh * Kw * Tm)
                B_out = (Tr * Tc * Tm) * S
            elif inter_lo == 'reuse_O':
                B_in = (Tri * Tci * Tm)
                B_w = (Kh * Kw * Tm)
                B_out = (Tr * Tc * Tm)
            else:
                sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")
        else:
            # Standard conv
            if inter_lo == 'reuse_I' or inter_lo == 'reuse_W':
                B_in = (Tri * Tci * Tn)
                B_w = (Kh * Kw * Tm * Tn)
                B_out = (Tr * Tc * Tm) * S
            elif inter_lo == 'reuse_O':
                B_in = (Tri * Tci * Tn)
                B_w = (Kh * Kw * Tm * Tn)
                B_out = (Tr * Tc * Tm)    
            else:
                sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")
    
    elif (layer_type == 'POOL'):
        B_in = (Tri * Tci * Tn)
        B_w = 0
        B_out = (Tr * Tc * Tm) * S
    
    elif  (layer_type == 'GAVGPOOL'):
        B_in = (Tri * Tci * Tm)
        B_w = 0  # no weights
        B_out = (1 * 1 * Tm) * S
        
    elif (layer_type == "BN"):
        B_in = (Tri * Tci * Tm) * S
        B_w = Tm*4  # mu, sigma, beta (weight), gamma (bias)
        B_out = 0  # inplace update does not need a separate buffer
        
    elif (layer_type == "RELU"):
        B_in = (Tri * Tci * Tm) * S
        B_w = 0  # no weights
        B_out = 0  # inplace update does not need a separate buffer
    
    elif (layer_type == "ADD"):
        B_in = (2 * Tri * Tci * Tm)     #Tm = Tn
        B_w = 0
        B_out = (Tr * Tc * Tm) * S
        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
    
    return B_in, B_w, B_out

#######################
# Parameter cleaning
#######################
# only CONV and FC may have reuse schemes
def filter_legal_reuseschems(layer_type = "CONV", op_type=OPTYPES.O_CONV2D):

    if (layer_type == 'CONV') or (layer_type == 'FC'):
        #inter_lo = ["reuse_I", "reuse_W", "reuse_O"]    
        if op_type == OPTYPES.O_CONV2D_DW or op_type == OPTYPES.O_CONV1D_DW:
            inter_lo = ["reuse_I", "reuse_W", "reuse_O"]    
        else:
            inter_lo = ["reuse_I", "reuse_W", "reuse_O"]    
        
    elif (layer_type == 'POOL') or (layer_type == 'GAVGPOOL'):
        inter_lo = ["reuse_O"]
    elif (layer_type == 'RELU') or (layer_type == 'ADD'):
        inter_lo = ["reuse_O"]    
    elif (layer_type == 'BN'):
        inter_lo = ["reuse_W"]
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
    return inter_lo
    




# each layer can have a valid number of tile sizes
# rules : 
# - tile dim should be a complete divisor of the layer dim
def filter_legal_tilesizes(lst_Tr, lst_Tc, lst_Tm, lst_Tn, H, W, R, C, M, N, layer_type = 'CONV', op_type=OPTYPES.O_CONV2D):

    if R % 2:
        upper_bound_R = R+1
    else:
        upper_bound_R = R

    if C % 2:
        upper_bound_C = C+1
    else:
        upper_bound_C = C

    if (layer_type == 'CONV') or (layer_type == 'FC'):
        Tr_step = 2; Tc_step = 2; Tn_step = 2; Tm_step = 2  # customizable step size
        
        if lst_Tr == None:    
            legal_lst_tr = [1] + [tr for tr in np.arange(2, upper_bound_R+1, Tr_step)]            
        else:
            legal_lst_tr = [tr for tr in lst_Tr if (upper_bound_R % tr) == 0]
        
        if lst_Tc == None:    
            legal_lst_tc = [1] + [tc for tc in np.arange(2, upper_bound_C+1, Tc_step)]            
        else:
            legal_lst_tc = [tc for tc in lst_Tc if (upper_bound_C % tc) == 0]
        
        if lst_Tm == None:    
            legal_lst_tm = [1] + [tm for tm in np.arange(2, M+1, Tm_step) if (M % tm) == 0]            
        else:
            legal_lst_tm = [tm for tm in lst_Tm if (M % tm) == 0]
        
        if op_type == OPTYPES.O_CONV2D_DW or op_type == OPTYPES.O_CONV1D_DW:
            legal_lst_tn = [1]  # Tm = Tn, so we ignore Tn
        else:
            if lst_Tn == None:    
                legal_lst_tn = [1] + [tn for tn in np.arange(2, N+1, Tn_step) if (N % tn) == 0]                        
            else:
                legal_lst_tn = [tn for tn in lst_Tn if (N % tn) == 0]    

    elif (layer_type == 'POOL'):
        Tr_step = 2; Tc_step = 2; Tn_step = 2; Tm_step = 2  # customizable step size
        if lst_Tr == None:    
            legal_lst_tr = [1] + [tr for tr in np.arange(2, upper_bound_R+1, Tr_step)]
        else:
            legal_lst_tr = [tr for tr in lst_Tr if (upper_bound_R % tr) == 0]
        
        if lst_Tc == None:    
            legal_lst_tc = [1] + [tc for tc in np.arange(2, upper_bound_C+1, Tc_step)]
        else:
            legal_lst_tc = [tc for tc in lst_Tc if (upper_bound_C % tc) == 0]
        
        if lst_Tm == None:    
            legal_lst_tm = [1] + [tm for tm in np.arange(2, M+1, Tm_step) if (M % tm) == 0]
        else:
            legal_lst_tm = [tm for tm in lst_Tm if (M % tm) == 0]
        
        # Tn is not a dimension for pool
        legal_lst_tn = [1]


    elif (layer_type == 'GAVGPOOL'):
        Tr_step = 2; Tc_step = 2; Tn_step = 2; Tm_step = 2  # customizable step size
        if lst_Tr == None:    
            legal_lst_tr = [1] + [tr for tr in np.arange(2, H+1, Tr_step) if (H % tr) == 0]
        else:
            legal_lst_tr = [tr for tr in lst_Tr if (H % tr) == 0]
        
        if lst_Tc == None:    
            legal_lst_tc = [1] + [tc for tc in np.arange(2, W+1, Tc_step) if (W % tc) == 0]
        else:
            legal_lst_tc = [tc for tc in lst_Tc if (W % tc) == 0]
        
        if lst_Tm == None:    
            legal_lst_tm = [1] + [tm for tm in np.arange(2, M+1, Tm_step) if (M % tm) == 0]
        else:
            legal_lst_tm = [tm for tm in lst_Tm if (M % tm) == 0]
        
        # Tn is not a dimension for pool
        legal_lst_tn = [1]
        
    
    elif (layer_type == "BN") or (layer_type == "RELU") or (layer_type == "ADD"):
        Tr_step = 2; Tc_step = 2; Tn_step = 2; Tm_step = 2  # customizable step size
        if lst_Tr == None:    
            legal_lst_tr = [1] + [tr for tr in np.arange(2, upper_bound_R+1, Tr_step)]
        else:
            legal_lst_tr = [tr for tr in lst_Tr if (upper_bound_R % tr) == 0]
        
        if lst_Tc == None:    
            legal_lst_tc = [1] + [tc for tc in np.arange(2, upper_bound_C+1, Tc_step)]
        else:
            legal_lst_tc = [tc for tc in lst_Tc if (upper_bound_C % tc) == 0]
        
        if lst_Tm == None:    
            legal_lst_tm = [1] + [tm for tm in np.arange(2, M+1, Tm_step) if (M % tm) == 0]
        else:
            legal_lst_tm = [tm for tm in lst_Tm if (M % tm) == 0]
          
        legal_lst_tn = [1]  # Tm = Tn, so we ignore Tn
                
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
    
    # should we remove 1 ?    
    #if len(legal_lst_tr)>1: legal_lst_tr.remove(1)
    #if len(legal_lst_tc)>1: legal_lst_tc.remove(1)
    #if len(legal_lst_tm)>1: legal_lst_tm.remove(1)
    #if len(legal_lst_tn)>1: legal_lst_tn.remove(1)
    legal_lst_tr=[R]
    legal_lst_tc=[C]
    legal_lst_tm=[M]
    legal_lst_tn=[N]  
        
        
    return legal_lst_tr, legal_lst_tc, legal_lst_tm, legal_lst_tn

 


# Unused for TINAS, as preSz=1 is always used
def filter_legal_pressizes(lst_S, H, W, R, C, M, N, Tr, Tc, Tm, Tn, inter_lo, layer_type = 'CONV'):
    if (layer_type == 'CONV') or (layer_type == 'FC'):
        if inter_lo == "reuse_I":        
            iters = int(np.ceil(M/Tm))
        elif inter_lo == "reuse_W":
            iters = int(np.ceil(R/Tr)*np.ceil(C/Tc))
        elif inter_lo == "reuse_O":
            iters = int(np.ceil(N/Tn))
        else:
            sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown inter-tile order")

    # assume only one loop order for pool: RCM
    elif layer_type == 'POOL':
        iters = int(np.ceil(M/Tm))    

    elif layer_type == 'GAVGPOOL':
        #iters = int(np.ceil(H/Tr)*np.ceil(W/Tc))
        iters = int(np.ceil(M/Tm))
        
    elif (layer_type == 'BN') or (layer_type == 'RELU') or (layer_type == 'ADD'):
        #iters = int(np.ceil(H/Tr)*np.ceil(W/Tc))
        iters = int(np.ceil(M/Tm))
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown layer type")
    

    # find legal S params for the iters
    if lst_S == None:
        legal_lst_S = [1] + [S for S in np.arange(2, iters+1, 2) if (iters % S) == 0]
    else:
        legal_lst_S = [S for S in lst_S if (iters % S) == 0]

    return legal_lst_S


#######################
# String Representations
#######################

def to_string_params_exec(pexec):
    return '_'.join([str(x) for x in pexec['tile_size']]) + '_' + pexec['inter_lo']
    
def to_string_params_pres(ppres):
    return str(ppres['backup_batch_size'])

def to_string_params_all(pexec, ppres):
    if pexec==None:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - pexec is None")
    elif ppres==None:
        s = '_'.join([str(x) for x in pexec['tile_size'][4:]]) + '_' + pexec['inter_lo']
    else:        
        s = '_'.join([str(x) for x in pexec['tile_size'][4:]]) + '_' + pexec['inter_lo'] + "_S" + str(ppres['backup_batch_size'])
    return s

def string_to_params_all(pstr):
    if "_S" in pstr:
        # eg: 2_2_1_1_reuse_I_S16
        Tr, Tc, Tm, Tn, tmp, reuse_sch, S = pstr.split("_")
        S = int(S.replace("S", ""))
        return int(Tr), int(Tc), int(Tm), int(Tn), tmp+"_"+reuse_sch, S
    else:
        # assume S = 1
        # eg: 2_2_1_1_reuse_I
        Tr, Tc, Tm, Tn, tmp, reuse_sch = pstr.split("_")        
        return int(Tr), int(Tc), int(Tm), int(Tn), tmp+"_"+reuse_sch, 1


#######################
# check for strange values
#######################
def check_infnan(vlst):
    for v in vlst:
        if np.isnan(v) or np.isinf(v):
            return False
        else:
            pass    
    return True
