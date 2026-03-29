import torch
import torch.nn as nn



'''
taken from : 
https://github.com/yukang2017/NAS-quantization


• ConvOP [0,1,2,3] conv/0, sep-conv/1, mobile-ib-conv-2/2, mobile-ib-conv-3/3, mobile-ib-conv-6/4
• KernelSize [0,1] 3x3/0, 5x5/1.
• SkipOp [0,1,2,3] max/0, avg/1, id/2, no/3.
• Filters  Fi ...  omit currently.
• Layers [0,1,2,3] 1/0, 2/1, 3/2, 4/3.
• Quantz [0,1,2]   4/0, 8/1, 16/2.
[ConvOP, KernelSize, SkipOp, Layers, Quantz]
'''

'''
OPS = {
    'conv': lambda kernel_size, skip_op, num_layers: Conv(kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers),
    'sep_conv': lambda kernel_size, skip_op, num_layers: SepConv(kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers),
    'mib_conv': lambda kernel_size, skip_op, num_layers: MobInvBConv(kernel_size=kernel_size, skip_op=skip_op, num_layers=num_layers),
}
'''
# MBConv = lambda name_prefix, C_in, C_out, kernel_size, expansion_factor, stride, padding, support_skip, affine: MBConv(name_prefix, 
#                                                                                                                        C_in, C_out, 
#                                                                                                                        kernel_size, stride, padding, 
#                                                                                                                        t=expansion_factor, 
#                                                                                                                        support_skip=support_skip, 
#                                                                                                                        affine=affine)

# MBConv1 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=1, support_skip=support_skip, affine=affine)
# MBConv2 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=2, support_skip=support_skip, affine=affine)
# MBConv3 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=3, support_skip=support_skip, affine=affine)
# MBConv4 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=4, support_skip=support_skip, affine=affine)
# MBConv5 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=5, support_skip=support_skip, affine=affine)
# MBConv6 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=6, support_skip=support_skip, affine=affine)
# MBConv7 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=7, support_skip=support_skip, affine=affine)
# MBConv8 = lambda name_prefix, C_in, C_out, kernel_size, stride, padding, support_skip, affine: MBConv(name_prefix, C_in, C_out, kernel_size, stride, padding, t=8, support_skip=support_skip, affine=affine)


# class Skip(nn.Module):
#     def __init__(self, C_in, C_out, skip_op, kernel_size, stride, padding, expansion_factor=None):
#         super(Skip, self).__init__()
#         if stride>1 or not C_in==C_out:
#             name = ''#'Skip-'
#             skip_conv = nn.Sequential()
#             skip_conv.add_module(name+'conv1',nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False))
#             skip_conv.add_module(name+'bn1',nn.BatchNorm2d(C_out, affine=True))
#             stride = 1
#             padding=int((kernel_size-1)/2)

#         if skip_op==0:
#             self.op=nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
#         elif skip_op==1:
#             self.op=nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
#         elif skip_op==2:
#             self.op=Identity()
#         elif skip_op==3:
#             self.op=Zero(stride)
#         else:
#             raise ValueError('Wrong skip_op {}'.format(skip_op))

#         if stride>1 or not C_in==C_out:
#             self.op=nn.Sequential(skip_conv,self.op)

#     def forward(self,x):
#         return self.op(x)


# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x):
#         return x


# class Zero(nn.Module):
#     def __init__(self, stride):
#         super(Zero, self).__init__()
#         self.stride = stride

#     def forward(self, x):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)

# class ReshapeTranspose(nn.Module):
#     def __init__(self, dim=1, exp=2):
#         super(ReshapeTranspose, self).__init__()
#         self.dim = dim
#         self.exp = exp

#         self.reshape1 = Reshape(0,dim,exp)  # Reshape is used with dynamic shape during forward
#         self.transpose = Transpose(1, 2)  # Initial transpose configuration
#         self.reshape2 = Reshape(1,dim,exp)

#     def forward(self, x, dim0=1, dim1=2):
#         # Step 1: Concatenate
#         # Step 2: Reshape dynamically with the provided shape
#         x = self.reshape1(x)
#         # Step 3: Transpose with specified dimensions
#         x = self.transpose(x)
#         # Step 4: Reshape to final shape
#         x = self.reshape2(x)
#         return x


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = 1  # Dimension to concatenate along

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


# class Reshape(nn.Module):
#     def __init__(self, mode, dim, exp):
#         super(Reshape, self).__init__()
#         self.mode = mode
#         self.dim = dim
#         self.exp = exp

#     def forward(self, x):
#         # Use the shape provided dynamically during the forward pass
#         batch_size, num_channels, height, width = x.size()
#         if self.mode == 0:
#             return x.reshape(batch_size, 2, num_channels//2, height, width)
#         else:
#             return x.reshape(batch_size, num_channels, height, width)


# class Transpose(nn.Module):
#     def __init__(self, dim0, dim1):
#         super(Transpose, self).__init__()
#         self.dim0 = dim0  # First dimension to swap
#         self.dim1 = dim1  # Second dimension to swap

#     def forward(self, x, dim0, dim1):
#         # Use the dimensions provided dynamically during the forward pass
#         return x.transpose(dim0, dim1)


class ShuffleNetV2Conv(nn.Module):
    conv_class = nn.Conv2d
    batchnorm_class = nn.BatchNorm2d

    def __init__(self, name_prefix, in_channels, out_channels, set_stride, set_kernel_size, batchnorm_epsilon=1e-5):
        super(ShuffleNetV2Conv, self).__init__()
        self.set_stride = set_stride

        name = 'shufflebk_'
        affine=True
        set_kernel_size = set_kernel_size
        mid_channels = out_channels // 2
        if set_kernel_size == 1:
            pads = 0
        else:
            pads = (set_kernel_size - 1) // 2

        self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential()
        # Using add_module instead of nn.Sequential
        if self.set_stride == 1:
            # Branch 1
            self.branch1.add_module(name+'branch1_conv1', self.conv_class(in_channels, out_channels-mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            self.branch1.add_module(name+'branch1_bn1', self.batchnorm_class(out_channels-mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch1.add_module(name+'branch1_relu1', nn.ReLU(inplace=True))

            # Branch 2            
            self.branch2.add_module(name+'branch2_conv1', self.conv_class(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            self.branch2.add_module(name+'branch2_bn1', self.batchnorm_class(mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch2.add_module(name+'branch2_relu1', nn.ReLU(inplace=True))
            self.branch2.add_module(name+'branch2_conv2', self.conv_class(mid_channels, mid_channels, kernel_size=set_kernel_size, stride=1, padding=pads, groups=mid_channels, bias=False))
            self.branch2.add_module(name+'branch2_bn2', self.batchnorm_class(mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch2.add_module(name+'branch2_conv3', self.conv_class(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            self.branch2.add_module(name+'branch2_bn3', self.batchnorm_class(mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch2.add_module(name+'branch2_relu2', nn.ReLU(inplace=True))

            
        else:
            # Branch 1 for stride=2
            
            self.branch1.add_module(name+'branch1_conv1', self.conv_class(in_channels, in_channels, kernel_size=set_kernel_size, stride=1, padding=pads, groups=in_channels, bias=False))
            self.branch1.add_module(name+'branch1_bn1', self.batchnorm_class(in_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch1.add_module(name+'branch1_relu1', nn.ReLU(inplace=True))

            self.branch1.add_module(name+'branch1_conv2', self.conv_class(in_channels, out_channels-mid_channels, kernel_size=1, stride=set_stride, padding=0, bias=False))
            self.branch1.add_module(name+'branch1_bn2' ,self.batchnorm_class(out_channels-mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch1.add_module(name+'branch1_relu2', nn.ReLU(inplace=True))


            # Branch 2 for stride=2
            
            self.branch2.add_module(name+'branch2_conv1', self.conv_class(in_channels, mid_channels, kernel_size=1, stride=set_stride, padding=0, bias=False))
            self.branch2.add_module(name+'branch2_bn1', self.batchnorm_class(mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch2.add_module(name+'branch2_relu1', nn.ReLU(inplace=True))

            self.branch2.add_module(name+'branch2_conv2', self.conv_class(mid_channels, mid_channels, kernel_size=set_kernel_size, stride=1, padding=pads, groups=mid_channels, bias=False))
            self.branch2.add_module(name+'branch2_bn2', self.batchnorm_class(mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch2.add_module(name+'branch2_conv3', self.conv_class(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            self.branch2.add_module(name+'branch2_bn3', self.batchnorm_class(mid_channels, affine=affine, eps=batchnorm_epsilon))
            self.branch2.add_module(name+'branch2_relu2', nn.ReLU(inplace=True))
            
            #self.concat = Concat(1)

        #self.concat = Concat(dim=1)
        # self.rtr = ReshapeTranspose(dim=1, exp=2)

        #self.reshape = Reshape()  # Custom dynamic reshape
        #self.transpose = Transpose(1,2)
    
    def forward(self, x):
        # Use the branches for stride = 1
        #print("x:", x.size())
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        #print("b1= ", branch1_out.size())
        #print("b2= ", branch2_out.size())
        #print("Branch 1 output shape:", branch1_out.size())  # For debugging
        #print("Branch 2 output shape:", branch2_out.size())  # For debugging

        out = torch.cat([branch1_out, branch2_out], 1)
        #print("out:", out.size())
        #out = concat(branch1_out, branch2_out)
        batch_size, num_channels, height, width = out.size()

        out = out.reshape(batch_size, 2, torch.div(num_channels, 2, rounding_mode='trunc'), height, width)  # Reshape
        #print("reshape:", out.size())
        out = out.transpose(1, 2)  # Transpose between dimensions 1 and 2
        out = out.reshape(batch_size, num_channels, height, width)  # Reshape back to desired size
        #print("reshape:", out.size())
        #out = concat(self.branch1(x), self.branch2(x))

        return out






    #     if self.set_stride == 1:
    #         self.branch1 = nn.Sequential(
    #             nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True),
    #         )
    #         self.branch2 = nn.Sequential(
    #             nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=pads, groups=mid_channels, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding = 0, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True),
    #         )
            
    #     else:
    #         print(in_channels, out_channels, mid_channels)
    #         # For stride = 2, both branches process the input
    #         self.branch1 = nn.Sequential(
    #             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
    #             nn.BatchNorm2d(in_channels),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True)
    #         )
    #         self.branch2 = nn.Sequential(
    #             nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #             nn.BatchNorm2d(mid_channels),
    #             nn.ReLU(inplace=True),
    #         )

    # def forward(self, x):
    #     if self.set_stride == 1:
    #         #x1, x2 = x.chunk(2, dim=1)  # Split channels
    #         out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    #     else:
    #         out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

    #     # Channel Shuffle
    #     batch_size, num_channels, height, width = out.size()
        
    #     out = out.reshape(batch_size, 2, num_channels // 2, height, width)
    #     out = out.permute(0, 2, 1, 3, 4).reshape(batch_size, num_channels, height, width)
    #     print("out of block: ", out.size())
    #     return out


# regular conv
class ReLUConvBN(nn.Module):
    def __init__(self, name_prefix, C_in, C_out, kernel_size, stride, padding, affine=True, expansion_factor=None):
        super(ReLUConvBN, self).__init__()
        name = name_prefix #'com'
        self.op = nn.Sequential()
        self.op.add_module(name+'relu',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv',nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False))
        self.op.add_module(name+'bn',nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, name_prefix, C_in, C_out, kernel_size, stride, padding, affine=True, expansion_factor=None):
        super(SepConv, self).__init__()
        name = name_prefix #'Sep-'
        self.op = nn.Sequential()
        self.op.add_module(name+'relu1',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv1',nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False))
        self.op.add_module(name+'conv2',nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False))
        self.op.add_module(name+'bn1',nn.BatchNorm2d(C_in, affine=affine))
        self.op.add_module(name+'relu2',nn.ReLU(inplace=False))
        self.op.add_module(name+'conv3',nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False))
        self.op.add_module(name+'conv4',nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False))
        self.op.add_module(name+'bn2',nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


# class MBConv(nn.Module):
#     conv_class = nn.Conv2d
#     batchnorm_class = nn.BatchNorm2d

#     def __init__(self, name_prefix, C_in, C_out, kernel_size, stride, padding, expansion_factor=3, support_skip=True, affine=True, batchnorm_epsilon=1e-5):
        
#         #print("C_in={}, C_out={}".format(C_in, C_out))
#         super(MBConv, self).__init__()
#         #name = name_prefix #'MB-'
#         name = 'mbconv_'
#         self.stride = stride
#         self.support_skip = support_skip
#         t = expansion_factor
        
#         self.op = nn.Sequential()
        
#         # expansion : pointwise (1x1) conv with BN
#         self.op.add_module(name+'conv0_pw', self.conv_class(C_in, C_in*t, kernel_size=1, stride=1, padding=0, groups=1, bias=False))
#         self.op.add_module(name+'bn0',self.batchnorm_class(C_in*t, affine=affine, eps=batchnorm_epsilon))
#         self.op.add_module(name+'relu0',nn.ReLU(inplace=False))

#         # depthwise conv with BN
#         self.op.add_module(name+'conv1_dw',self.conv_class(C_in*t, C_in*t, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in*t, bias=False))
#         self.op.add_module(name+'bn1',self.batchnorm_class(C_in*t, affine=affine, eps=batchnorm_epsilon))
#         self.op.add_module(name+'relu1',nn.ReLU(inplace=False))
        
#         # projection : pointwise (1x1) conv with BN
#         self.op.add_module(name+'conv2_pw',self.conv_class(C_in*t, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False))
#         self.op.add_module(name+'bn2',self.batchnorm_class(C_out, affine=affine, eps=batchnorm_epsilon))
        
#         if (self.support_skip):
#             # skip connection (only if stride==1)
#             if self.stride == 1: 
#                 self.shortcut = nn.Sequential()        # identity stride
#                 self.shortcut.add_module(name+"skip_identity", nn.Identity())
                
#                 if stride == 1 and (C_in != C_out):    # if dimensions are incompatible, then need to use 1x1 conv to fix
#                     self.shortcut = nn.Sequential() 
#                     self.shortcut.add_module(name+"skip_conv3_pw", self.conv_class(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False))
#                     self.shortcut.add_module(name+"skip_bn3", self.batchnorm_class(C_out, eps=batchnorm_epsilon))
            
#             else:
#                 self.shortcut = None
#         else:
#             self.shortcut = None
                
            
        

#     def forward(self, x):                        
#         out = self.op(x)        
#         if (self.support_skip):
#             out = (out + self.shortcut(x)) if self.stride==1 else out                
#         return out
        
    
    
#     # def _get_layer_config(self):
        
#     #     config = []
#     #     for each_op in self.op:
#     #         if isinstance(each_op, nn.Conv2d):
#     #         elif isinstance(each_op, nn.BatchNorm2d):
#     #         elif isinstance(each_op, nn.ReLU):
                
    

# class MBConv1D(MBConv):
#     conv_class = nn.Conv1d
#     batchnorm_class = nn.BatchNorm1d
