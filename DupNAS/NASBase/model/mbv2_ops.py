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


class MBConv(nn.Module):
    conv_class = nn.Conv2d
    batchnorm_class = nn.BatchNorm2d

    def __init__(self, name_prefix, C_in, C_out, kernel_size, stride, padding, expansion_factor=4, support_skip=True, affine=True, batchnorm_epsilon=1e-5):
        
        #print("C_in={}, C_out={}".format(C_in, C_out))
        super(MBConv, self).__init__()
        #name = name_prefix #'MB-'
        name = 'mbv2_'
        self.stride = stride
        t = expansion_factor
        self.firstop = nn.Sequential()
        self.op = nn.Sequential()
        #self.shortcut = nn.Sequential() 
        
        
        
        # expansion : pointwise (1x1) conv with BN
        self.firstop.add_module(name+'conv0_pw', self.conv_class(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False))
        self.firstop.add_module(name+'bn0',self.batchnorm_class(C_in, affine=affine, eps=batchnorm_epsilon))
        self.firstop.add_module(name+'relu0',nn.ReLU(inplace=False))

        self.firstop.add_module(name+'conv1_pw', self.conv_class(C_in, C_out//2, kernel_size=1, stride=1, padding=0, bias=False))
        self.firstop.add_module(name+'bn1',self.batchnorm_class(C_out//2, affine=affine, eps=batchnorm_epsilon))
        self.firstop.add_module(name+'relu1',nn.ReLU(inplace=False))


        # expansion : pointwise (1x1) conv with BN
        self.op.add_module(name+'branch1-conv0_pw', self.conv_class(C_out//2, t*C_in, kernel_size=1, stride=1, padding=0, groups=1, bias=False))
        self.op.add_module(name+'branch1-bn0',self.batchnorm_class(t*C_in, affine=affine, eps=batchnorm_epsilon))
        self.op.add_module(name+'branch1-relu0',nn.ReLU(inplace=False))

        # depthwise conv with BN
        self.op.add_module(name+'branch1-conv1_dw',self.conv_class(t*C_in, t*C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=t*C_in, bias=False))
        self.op.add_module(name+'branch1-bn1',self.batchnorm_class(t*C_in, affine=affine, eps=batchnorm_epsilon))
        self.op.add_module(name+'branch1-relu1',nn.ReLU(inplace=False))
        
        # projection : pointwise (1x1) conv with BN
        self.op.add_module(name+'branch1-conv2_pw',self.conv_class(t*C_in, C_out//2, kernel_size=1, stride=1, padding=0, groups=1, bias=False))
        self.op.add_module(name+'branch1-bn2',self.batchnorm_class(C_out//2, affine=affine, eps=batchnorm_epsilon))
        
        #if self.stride == 1:
        #    self.shortcut = nn.Sequential() 
        #    self.shortcut.add_module(name+"branch2-conv3_pw", self.conv_class(C_in, C_out*2, kernel_size=1, stride=1, padding=0, bias=False))
        #    self.shortcut.add_module(name+"branch2-bn3", self.batchnorm_class(C_out*2, eps=batchnorm_epsilon))

        # if (self.support_skip) and (stride == 1):
        #     # skip connection (only if stride==1)
        #     if C_in == C_out*2: 
        #         self.shortcut = nn.Sequential()        # identity stride
        #         self.shortcut.add_module(name+"skip_identity", nn.Identity())
                
        #     else:    # if dimensions are incompatible, then need to use 1x1 conv to fix
        #         self.shortcut = nn.Sequential() 
        #         self.shortcut.add_module(name+"skip_conv3_pw", self.conv_class(C_in, C_out*2, kernel_size=1, stride=1, padding=0, bias=False))
        #         self.shortcut.add_module(name+"skip_bn3", self.batchnorm_class(C_out*2, eps=batchnorm_epsilon))
            
        # else:
        #     self.shortcut = None
        
        self.op2nd = nn.Sequential()
        
        # expansion : pointwise (1x1) conv with BN
        self.op2nd.add_module(name+'conv4_pw', self.conv_class(C_out//2, C_out, kernel_size=1, stride=1, padding=0, bias=False))
        self.op2nd.add_module(name+'bn4',self.batchnorm_class(C_out, affine=affine, eps=batchnorm_epsilon))
        self.op2nd.add_module(name+'relu4',nn.ReLU(inplace=False))
                
            
        

    def forward(self, x):
        a = self.firstop(x)                 
        b = self.op(a) 
        # if self.stride == 1:
            # tmp2 = self.shortcut(x)
        out = self.op2nd(a+b) 
        # else:
            # out=self.op2nd(tmp1) 

        # if (self.support_skip):
        #     tmp = (out + self.shortcut(x)) if self.stride==1 else out        
        # out = self.op2nd(tmp)       
        return out
        
    
    
    # def _get_layer_config(self):
        
    #     config = []
    #     for each_op in self.op:
    #         if isinstance(each_op, nn.Conv2d):
    #         elif isinstance(each_op, nn.BatchNorm2d):
    #         elif isinstance(each_op, nn.ReLU):
                
    

class MBConv1D(MBConv):
    conv_class = nn.Conv1d
    batchnorm_class = nn.BatchNorm1d
