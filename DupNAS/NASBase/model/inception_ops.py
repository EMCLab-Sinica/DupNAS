import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

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


class InceptionA(nn.Module):
    conv_class = nn.Conv2d
    batchnorm_class = nn.BatchNorm2d
    pooling_class = nn.AvgPool2d
    
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        # Quarter the number of output channels
        name = 'inception_'

        branch_channels = output_channels // 4
        pads = (kernel_size-1) // 2
        batchnorm_epsilon =1e-5 
        affine = True

        self.branch1x1 = nn.Sequential()
        self.branch5x5 = nn.Sequential()
        self.branch3x3 = nn.Sequential()
        self.branchpool = nn.Sequential()
        self.seq_add_module = nn.Sequential()

        self.branch1x1.add_module(name+'br1_conv0_pw',self.conv_class(input_channels, output_channels-3*branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        self.branch1x1.add_module(name+'br1_bn0',self.batchnorm_class(output_channels-3*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch1x1.add_module(name+'br1_relu0',nn.ReLU(inplace=False))

        self.branch5x5.add_module(name+'br2_conv1_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        self.branch5x5.add_module(name+'br2_bn1',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch5x5.add_module(name+'br2_relu1',nn.ReLU(inplace=False))
        self.branch5x5.add_module(name+'br2_conv2',self.conv_class(branch_channels, branch_channels, kernel_size=kernel_size, stride=1, padding=pads, bias=False))
        self.branch5x5.add_module(name+'br2_bn2',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch5x5.add_module(name+'br2_relu2',nn.ReLU(inplace=False))

        self.branch3x3.add_module(name+'br3_conv3_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        self.branch3x3.add_module(name+'br3_bn3',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3.add_module(name+'br3_relu3',nn.ReLU(inplace=False))
        self.branch3x3.add_module(name+'br3_conv3',self.conv_class(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.branch3x3.add_module(name+'br3_bn3',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3.add_module(name+'br3_relu3',nn.ReLU(inplace=False))

        self.branchpool.add_module(name+'br4_pool4',self.pooling_class(kernel_size=3, stride=1, padding=1))
        self.branchpool.add_module(name+'br4_conv5_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride))
        self.branchpool.add_module(name+'br4_bn5',self.batchnorm_class(branch_channels, eps=batchnorm_epsilon))
        self.branchpool.add_module(name+'br4_relu5',nn.ReLU(inplace=False))

        self.seq_add_module.add_module(name+'seq_conv6_pw',self.conv_class(output_channels, output_channels, kernel_size=1, stride=1, groups=1, bias=False))
        #self.branch1x1 = BasicConv2d(input_channels, output_channels-3*branch_channels, stride=stride ,kernel_size=1)

        # self.branch5x5 = nn.Sequential(
        #     BasicConv2d(input_channels, branch_channels, kernel_size=1),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride, padding=pads)
        # )

        # self.branch3x3 = nn.Sequential(
        #     BasicConv2d(input_channels, branch_channels, kernel_size=1),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=3, stride=stride, padding=1)
        # )

        # self.branchpool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        #     BasicConv2d(input_channels, branch_channels, stride=stride, kernel_size=1)
        # )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branchpool = self.branchpool(x)
        mergebr = torch.cat([branch1x1, branch5x5, branch3x3, branchpool], 1)
        outputs = self.seq_add_module(mergebr)
        return outputs



class InceptionB(nn.Module):
    conv_class = nn.Conv2d
    batchnorm_class = nn.BatchNorm2d
    pooling_class = nn.AvgPool2d

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        # Quarter the output channels
        name = 'inception_'

        branch_channels = output_channels // 4
        pads = (kernel_size-1) // 2
        batchnorm_epsilon =1e-5 
        affine = True

        self.branch1x1 = nn.Sequential()
        self.branch7x7 = nn.Sequential()
        self.branch7x7stack = nn.Sequential()
        self.branchpool = nn.Sequential()
        self.seq_add_module = nn.Sequential()

        self.branch1x1.add_module(name+'br1_conv0_pw',self.conv_class(input_channels, output_channels-3*branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        self.branch1x1.add_module(name+'br1_bn0',self.batchnorm_class(output_channels-3*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch1x1.add_module(name+'br1_relu0',nn.ReLU(inplace=False))

        self.branch7x7.add_module(name+'br2_conv1_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        self.branch7x7.add_module(name+'br2_bn1',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7.add_module(name+'br2_relu1',nn.ReLU(inplace=False))
        self.branch7x7.add_module(name+'br2_conv2',self.conv_class(branch_channels, branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0), bias=False))
        self.branch7x7.add_module(name+'br2_conv3',self.conv_class(branch_channels, branch_channels, kernel_size=(1, kernel_size), padding=(0, pads), bias=False))
        self.branch7x7.add_module(name+'br2_bn3',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7.add_module(name+'br2_relu3',nn.ReLU(inplace=False))
        # self.branch7x7.add_module(name+'br2_conv4_pw',self.conv_class(branch_channels, branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        # self.branch7x7.add_module(name+'br2_bn4',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        # self.branch7x7.add_module(name+'br2_relu4',nn.ReLU(inplace=False))

        self.branch7x7stack.add_module(name+'br3_conv5_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride, groups=1, bias=False))
        self.branch7x7stack.add_module(name+'br3_bn5',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7stack.add_module(name+'br3_relu5',nn.ReLU(inplace=False))
        self.branch7x7stack.add_module(name+'br3_conv6',self.conv_class(branch_channels, branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0), bias=False))
        self.branch7x7stack.add_module(name+'br3_bn6',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7stack.add_module(name+'br3_relu6',nn.ReLU(inplace=False))
        self.branch7x7stack.add_module(name+'br3_conv7',self.conv_class(branch_channels, branch_channels, kernel_size=(1, kernel_size), padding=(0, pads), bias=False))
        self.branch7x7stack.add_module(name+'br3_bn7',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7stack.add_module(name+'br3_relu7',nn.ReLU(inplace=False))
        self.branch7x7stack.add_module(name+'br3_conv8',self.conv_class(branch_channels, branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0), bias=False))
        self.branch7x7stack.add_module(name+'br3_bn8',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7stack.add_module(name+'br3_relu8',nn.ReLU(inplace=False))
        self.branch7x7stack.add_module(name+'br3_conv9',self.conv_class(branch_channels, branch_channels, kernel_size=(1, kernel_size), padding=(0, pads), bias=False))
        self.branch7x7stack.add_module(name+'br3_bn9',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch7x7stack.add_module(name+'br3_relu9',nn.ReLU(inplace=False))

        self.branchpool.add_module(name+'br4_pool10',self.pooling_class(kernel_size=3, stride=1, padding=1))
        self.branchpool.add_module(name+'br4_conv11_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride))
        self.branchpool.add_module(name+'br4_bn11',self.batchnorm_class(branch_channels, eps=batchnorm_epsilon))
        self.branchpool.add_module(name+'br4_relu11',nn.ReLU(inplace=False))

        self.seq_add_module.add_module(name+'seq_conv12_pw',self.conv_class(output_channels, output_channels, kernel_size=1, stride=1, groups=1, bias=False))
        


        # self.branch1x1 = BasicConv2d(input_channels, output_channels-3*branch_channels, kernel_size=1, stride=stride)

        # self.branch7x7 = nn.Sequential(
        #     BasicConv2d(input_channels, branch_channels, kernel_size=1),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0)),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=(1, kernel_size), padding=(0, pads)),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=1, stride=stride)
        # )

        # self.branch7x7stack = nn.Sequential(
        #     BasicConv2d(input_channels, branch_channels, kernel_size=1, stride=stride),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0)),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=(1, kernel_size), padding=(0, pads)),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0)),
        #     BasicConv2d(branch_channels, branch_channels, kernel_size=(1, kernel_size), padding=(0, pads))
        # )

        # self.branchpool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        #     BasicConv2d(input_channels, branch_channels, kernel_size=1, stride=stride),
        # )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7stack = self.branch7x7stack(x)
        branchpool = self.branchpool(x)
        mergebr = torch.cat([branch1x1, branch7x7, branch7x7stack, branchpool], 1)
        outputs = self.seq_add_module(mergebr)
        return outputs

class InceptionC(nn.Module):
    conv_class = nn.Conv2d
    batchnorm_class = nn.BatchNorm2d
    pooling_class = nn.AvgPool2d

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        # Quarter the output channels
        name = 'inception_'
        
        branch_channels = output_channels // 10
        pads = (kernel_size-1) // 2
        batchnorm_epsilon =1e-5 
        affine = True

        self.branch1x1 = nn.Sequential()
        self.branch3x3_1 = nn.Sequential()
        self.branch3x3_2a = nn.Sequential()
        self.branch3x3_2b = nn.Sequential()
        self.branch3x3stack_1 = nn.Sequential()
        self.branch3x3stack_3a = nn.Sequential()
        self.branch3x3stack_3b = nn.Sequential()
        self.branchpool = nn.Sequential()
        self.seq_add_module = nn.Sequential()

        self.branch1x1.add_module(name+'br1_conv0_pw',self.conv_class(input_channels, output_channels-9*branch_channels, kernel_size=1, stride=stride, bias=False))
        self.branch1x1.add_module(name+'br1_bn0',self.batchnorm_class(output_channels-9*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch1x1.add_module(name+'br1_relu0',nn.ReLU(inplace=False))

        self.branch3x3_1.add_module(name+'br2_conv1_pw',self.conv_class(input_channels, 2*branch_channels, kernel_size=1, stride=stride, bias=False))
        self.branch3x3_1.add_module(name+'br2_bn1',self.batchnorm_class(2*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3_1.add_module(name+'br2_relu1',nn.ReLU(inplace=False))
        
        self.branch3x3_2a.add_module(name+'br2_conv2',self.conv_class(2*branch_channels, 2*branch_channels, kernel_size=(1, kernel_size), padding=(0, pads), stride=1, bias=False))
        self.branch3x3_2a.add_module(name+'br2_bn2',self.batchnorm_class(2*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3_2a.add_module(name+'br2_relu2',nn.ReLU(inplace=False))
        self.branch3x3_2b.add_module(name+'br2_conv3',self.conv_class(2*branch_channels, 2*branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0), stride=1, bias=False))
        self.branch3x3_2b.add_module(name+'br2_bn3',self.batchnorm_class(2*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3_2b.add_module(name+'br2_relu3',nn.ReLU(inplace=False))


        self.branch3x3stack_1.add_module(name+'br3_conv4_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride, padding=0, bias=False))
        self.branch3x3stack_1.add_module(name+'br3_bn4',self.batchnorm_class(branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3stack_1.add_module(name+'br3_relu4',nn.ReLU(inplace=False))
        self.branch3x3stack_1.add_module(name+'br3_conv5',self.conv_class(branch_channels, 2*branch_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.branch3x3stack_1.add_module(name+'br3_bn5',self.batchnorm_class(2*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3stack_1.add_module(name+'br3_relu5',nn.ReLU(inplace=False))

        self.branch3x3stack_3a.add_module(name+'br3_conv6',self.conv_class(2*branch_channels, 2*branch_channels, kernel_size=(1, kernel_size), padding=(0, pads), stride=1, bias=False))
        self.branch3x3stack_3a.add_module(name+'br3_bn6',self.batchnorm_class(2*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3stack_3a.add_module(name+'br3_relu6',nn.ReLU(inplace=False))
        self.branch3x3stack_3b.add_module(name+'br3_conv7',self.conv_class(2*branch_channels, 2*branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0), stride=1, bias=False))
        self.branch3x3stack_3b.add_module(name+'br3_bn7',self.batchnorm_class(2*branch_channels, affine=affine, eps=batchnorm_epsilon))
        self.branch3x3stack_3b.add_module(name+'br3_relu7',nn.ReLU(inplace=False))

        self.branchpool.add_module(name+'br4_pool10',self.pooling_class(kernel_size=3, stride=1, padding=1))
        self.branchpool.add_module(name+'br4_conv11_pw',self.conv_class(input_channels, branch_channels, kernel_size=1, stride=stride))
        self.branchpool.add_module(name+'br4_bn11',self.batchnorm_class(branch_channels, eps=batchnorm_epsilon))
        self.branchpool.add_module(name+'br4_relu11',nn.ReLU(inplace=False))


        self.seq_add_module.add_module(name+'seq_conv12_pw',self.conv_class(output_channels, output_channels, kernel_size=1, stride=1, groups=1, bias=False))
        

        # self.branch1x1 = BasicConv2d(input_channels, output_channels-9*branch_channels, kernel_size=1, stride=stride)

        # self.branch3x3_1 = BasicConv2d(input_channels, 2*branch_channels, kernel_size=1, stride=stride)
        # self.branch3x3_2a = BasicConv2d(2*branch_channels, 2*branch_channels, kernel_size=(1, kernel_size), padding=(0, pads))
        # self.branch3x3_2b = BasicConv2d(2*branch_channels, 2*branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0))

        # self.branch3x3stack_1 = BasicConv2d(input_channels, branch_channels, kernel_size=1)
        # self.branch3x3stack_2 = BasicConv2d(branch_channels, 2*branch_channels, kernel_size=3, stride=stride, padding=1)
        # self.branch3x3stack_3a = BasicConv2d(2*branch_channels, 2*branch_channels, kernel_size=(1, kernel_size), padding=(0, pads))
        # self.branch3x3stack_3b = BasicConv2d(2*branch_channels, 2*branch_channels, kernel_size=(kernel_size, 1), padding=(pads, 0))

        # self.branch_pool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        #     BasicConv2d(input_channels, branch_channels, kernel_size=1, stride=stride)
        # )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3_br1 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3_br1, 1)

        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack_br2 = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack_br2, 1)

        branchpool = self.branchpool(x)
        mergebr = torch.cat([branch1x1, branch3x3, branch3x3stack, branchpool],1)
        outputs = self.seq_add_module(mergebr)
        return outputs

