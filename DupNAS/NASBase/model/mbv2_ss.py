# ------- Search space ----------


'''
- First layer is fixed: CONV 3x3
- certain parameters are proportions of what is used in mobilenetV2: num_out_channels, num_layers

* ConvOP [0,1,2,3] conv/0, sep-conv/1, mobile-ib-conv-3/2, mobile-ib-conv-6/3.
* KernelSize [0,1] 3x3/0, 5x5/1.
* SkipOp [0,1,2,3] max/0, avg/1, id/2, no/3.
* Filters  Fi ...  omit currently.
* Layers [0,1,2,3] 1/0, 2/1, 3/2, 4/3.
* Quantz [0,1,2]   4/0, 8/1, 16/2.
[ConvOP, KernelSize, SkipOp, Layers, Quantz]

'''

# ==========================================================================================================================
# ================================= Common settings ========================================================================
# ==========================================================================================================================

FIRST_BLOCK_EXP_FACTOR = 1

# ==========================================================================================================================
# ====================================== IMAGE 100 ==========================================================================
# ==========================================================================================================================
EXP_FACTORS_IMAGE100 = [3, 4, 6]  #2,3,4,6
KERNEL_SIZES_IMAGE100 = [3, 5]  #1,3,5,7
MOBILENET_NUM_LAYERS_EXPLICIT_IMAGE100 = [1, 2, 3]
SUPPORT_SKIP_IMAGE100 = [True]  #set it true 
    
WIDTH_MULTIPLIER_IMAGE100 = [0.25, 0.5, 0.75, 1.0]
INPUT_RESOLUTION_IMAGE100 = [32, 64, 96, 128]
    
MOBILENET_V2_NUM_OUT_CHANNELS_IMAGE100 = [48, 96, 192, 384] #16, 24, 32, 64, 96, 160, 320

# ==========================================================================================================================
# ====================================== CIFAR 10 ==========================================================================
# ==========================================================================================================================

EXP_FACTORS_CIFAR10 = [1, 3, 4, 6] # mobile-ib-conv-2/2, mobile-ib-conv-3/3, mobile-ib-conv-6/4
KERNEL_SIZES_CIFAR10 = [1, 3, 5, 7]
#SE_RATIOS= {0:0.0, 1:0.25} # TODO: not implemented yet!
#SKIP_OPS_CIFAR10 = {0:'max_pool', 1:'avg_pool', 2:'identity', 3:'no_skip'}
MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10 = [1, 2, 3] # also called "repeat" in mbnet_v2 paper
SUPPORT_SKIP_CIFAR10 = [False, True]  # support skip in block

#STRIDE_FIRST_CIFAR10 = [1, 2] # stride for the first layer of each block (determines reduction)


NUM_LAYERS_MBNET_DELTA_CIFAR10 = [-1, 0, +1]  # with respect to the above mobilenetv2 num layers
NUM_OUT_CHANNELS_MBNET_RATIO_CIFAR10 = [0.75, 1.0, 1.25] # with respect to the above mobilenetv2 channels

NUM_BLOCKS_CIFAR10 = [3, 4, 5] # 4 by default


WIDTH_MULTIPLIER_CIFAR10 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INPUT_RESOLUTION_CIFAR10 = [32, 28]

MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10 = [16, 24, 32, 64, 96, 160, 320] # num out channels per block <-- hardcoded, not part of SS

# [16, 24, 32, 64, 96, 160, 320] <-- 1.0
# [8, 12, 16, 32, 48, 80, 160] <-- 0.5
# [12, 18, 24, 48, 72, 120, 240] <-- 0.75
# [20, 30, 40, 80, 120, 200, 400] <-- 1.25

#MOBILENET_V2_NUM_LAYERS_CIFAR10 = [1, 2, 3, 4, 3, 3, 1] # ORIGINAL num layers per block in mobilenetv2
# MOBILENET_V2_NUM_LAYERS_CIFAR10 = [1, 1, 2, 3, 2, 1, 1] # num layers per block 

# ==========================================================================================================================
# ====================================== HAR ===============================================================================
# ==========================================================================================================================

# reference: In https://github.com/leonardloh/MobileNet-SVM-HAR/,
# WIDTH_MULTIPLIER=1.0, EXP_FACTOR=1, KERNEL_SIZE=3, NUM_LAYERS_EXPLICIT=1, MOBILENET_V2_NUM_OUT_CHANNELS = [4, 8, 16, 32, 64, 128, 256]

EXP_FACTORS_HAR = [1, 3, 4, 6] # mobile-ib-conv-2/2, mobile-ib-conv-3/3, mobile-ib-conv-6/4
KERNEL_SIZES_HAR = [1, 3, 5, 7]
SUPPORT_SKIP_HAR = [False, True]  # support skip in block
STRIDE_FIRST_HAR = [1, 2] # stride for the first layer of each block (determines reduction)
MOBILENET_NUM_LAYERS_EXPLICIT_HAR = [1, 2, 3] # also called "repeat" in mbnet_v2 paper

# Not using very small width multipliers, otherwise the minimum possible output channel = 0.2*8 ~= 2 is too small
WIDTH_MULTIPLIER_HAR = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INPUT_RESOLUTION_HAR = [64, 128] 

MOBILENET_V2_NUM_OUT_CHANNELS_HAR = [8, 12, 16, 32, 48, 80, 160] # num out channels per block <-- hardcoded, not part of SS, half of CIFAR's channels
NUM_BLOCKS_HAR = [2, 3] # 3 by default

# ==========================================================================================================================
# ====================================== KWS [@TODO: not yet implemented] ==================================================
# ==========================================================================================================================

EXP_FACTORS_KWS = [1, 3] # mobile-ib-conv-2/2, mobile-ib-conv-3/3, mobile-ib-conv-6/4
KERNEL_SIZES_KWS = [1, 3]
SUPPORT_SKIP_KWS = [False, True]  # support skip in block
STRIDE_FIRST_KWS = [1, 2] # stride for the first layer of each block (determines reduction)
MOBILENET_NUM_LAYERS_EXPLICIT_KWS = [1, 2] # also called "repeat" in mbnet_v2 paper

WIDTH_MULTIPLIER_KWS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INPUT_RESOLUTION_KWS = [125, 250] 

MOBILENET_V2_NUM_OUT_CHANNELS_KWS = [4, 8, 16, 32, 64, 128, 256] # num out channels per block <-- hardcoded, not part of SS, half of CIFAR's channels

NUM_BLOCKS_KWS = [2, 3] # 3 by default
