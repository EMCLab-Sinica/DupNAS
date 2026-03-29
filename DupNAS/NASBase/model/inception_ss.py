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

INCEPT_STRIDE_FACTORS_IMAGE100 = [1, 2] 
INCEPT_KERNEL_SIZES_IMAGE100 = [3, 5, 7]
INCEPT_MODULE_ABC_IMAGE100 = [1] #module A only: 1, module ABC:123
INCEPT_NUM_LAYERS_EXPLICIT_IMAGE100 = [1, 2, 3] # also called "repeat" in mbnet_v2 paper

INCEPT_WIDTH_MULTIPLIER_IMAGE100 = [0.25, 0.5, 0.75, 1.0] # width of out channel: WIDTH_MULTIPLIER* NUM_OUT_CHANNELS
INCEPT_INPUT_RESOLUTION_IMAGE100 = [32, 64, 96, 128]
INCEPT_NUM_OUT_CHANNELS_IMAGE100 = [48, 96, 384, 640] # num out channels per block <-- hardcoded, not part of SS


# ==========================================================================================================================
# ====================================== CIFAR 10 ==========================================================================
# ==========================================================================================================================

#### parameters for incepotion block ####
# Assume a inception block contains 3 or 4 paths:
# path 0 for one 1x1 conv 
# path 1 and path 2 are selectable
# path 3 for one pooling + one 1x1 conv
#EXP_FACTORS_CIFAR10 = [1, 3, 4, 6] # mobile-ib-conv-2/2, mobile-ib-conv-3/3, mobile-ib-conv-6/4

INCEPT_STRIDE_FACTORS_CIFAR10 = [1, 2] 
INCEPT_KERNEL_SIZES_CIFAR10 = [3, 5, 7]
INCEPT_MODULE_ABC_CIFAR10 = [1,2,3]
INCEPT_NUM_LAYERS_EXPLICIT_CIFAR10 = [1, 2, 3] # also called "repeat" in mbnet_v2 paper

#### parameters for supernet ####
#choise for mbnet#
INCEPT_WIDTH_MULTIPLIER_CIFAR10 = [0.1, 0.2, 0.4, 0.5, 0.6] # width of out channel: WIDTH_MULTIPLIER* NUM_OUT_CHANNELS
INCEPT_INPUT_RESOLUTION_CIFAR10 = [64,96,128] #32, 64, 96,
INCEPT_NUM_OUT_CHANNELS_CIFAR10 = [96, 384, 640] # num out channels per block <-- hardcoded, not part of SS

##choise for incepotion#  #0.2, 0.4,
##0.2, 0.4, 0.5, 0.6, 0.8, 1.0 

STRIDE_FIRST_CIFAR10 = [1, 2] # stride for the first layer of each block (determines reduction)

NUM_LAYERS_MBNET_DELTA_CIFAR10 = [-1, 0, +1]  # with respect to the above mobilenetv2 num layers
NUM_OUT_CHANNELS_MBNET_RATIO_CIFAR10 = [0.75, 1.0, 1.25] # with respect to the above mobilenetv2 channels

NUM_BLOCKS_CIFAR10 = [3] # 4 by default  #3, 4, 5


#[16, 24, 32, 64, 96, 160, 320] <-- 1.0
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
