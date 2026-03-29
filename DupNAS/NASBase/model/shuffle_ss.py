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
# ====================================== IMAGE100 ==========================================================================
# ==========================================================================================================================
# IMAGE100
SHUFFLE_STRIDE_FACTORS_IMAGE100  = [1, 2] 
SHUFFLE_KERNEL_SIZES_IMAGE100 = [1, 3, 5, 7]
SHUFFLE_NUM_LAYERS_EXPLICIT_IMAGE100 = [1, 2, 3]

SHUFFLE_WIDTH_MULTIPLIER_IMAGE100 = [0.2, 0.5, 0.8] #0.2, 0.5,0.8
SHUFFLE_INPUT_RESOLUTION_IMAGE100 = [32, 64, 96, 128]
SHUFFLE_NUM_OUT_CHANNELS_IMAGE100 = [48, 96, 192, 384]

# ==========================================================================================================================
# ====================================== CIFAR 10 ==========================================================================
# ==========================================================================================================================

SHUFFLE_STRIDE_FACTORS_CIFAR10 = [1, 2] 
SHUFFLE_KERNEL_SIZES_CIFAR10 = [1, 3, 5, 7]
SHUFFLE_NUM_LAYERS_EXPLICIT_CIFAR10 = [1, 2, 3] # also called "repeat" in mbnet_v2 paper

#SUPPORT_SKIP_CIFAR10 = [False, True]  # support skip in block
STRIDE_FIRST_CIFAR10 = [1, 2] # stride for the first layer of each block (determines reduction)


NUM_LAYERS_MBNET_DELTA_CIFAR10 = [-1, 0, +1]  # with respect to the above mobilenetv2 num layers
NUM_OUT_CHANNELS_MBNET_RATIO_CIFAR10 = [0.75, 1.0, 1.25] # with respect to the above mobilenetv2 channels

NUM_BLOCKS_CIFAR10 = [3, 4, 5] # 4 by default


SHUFFLE_WIDTH_MULTIPLIER_CIFAR10 = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
SHUFFLE_INPUT_RESOLUTION_CIFAR10 = [64, 96, 128]

SHUFFLE_NUM_OUT_CHANNELS_CIFAR10 = [48, 96, 192, 384] # num out channels per block <-- hardcoded, not part of SS

# [16, 24, 32, 64, 96, 160, 320] <-- 1.0
# [8, 12, 16, 32, 48, 80, 160] <-- 0.5
# [12, 18, 24, 48, 72, 120, 240] <-- 0.75
# [20, 30, 40, 80, 120, 200, 400] <-- 1.25

