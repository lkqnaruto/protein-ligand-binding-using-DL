import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .utils import *
import sys

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=8,                # in_planes = 24, out_planes = 24, kernel_size = 55, groups = 6, stride = 1, width = True
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes                  #   24
        self.out_planes = out_planes                #   24
        self.groups = groups                        #   6
        self.group_planes = out_planes // groups    #   4
        self.kernel_size = kernel_size              #   55
        self.stride = stride                        #   1
        self.bias = bias                            #   False
        self.width = width                          #   True

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)

        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))





class AxialBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None, groups=1,              # inplanes=24, planes=24, kernel_size=55, stride=1, downsample = None, groups = 6
                 base_width=32, dilation=1, norm_layer=None):                                           # base_width = 64, dilation=1, norm_layer = BatchNorm2d
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes                                              # width = 24

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)                   # inplanes = 24, width = 24
        self.bn1 = norm_layer(width)
        # self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)  # width = 24, groups = 6, kernel_size=55, stride = 1, width
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x             # torch.Size([8, 32, 56, 56])
        out = self.conv_down(x)                # torch.Size([8, 64, 56, 56])
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.hight_block(out)                 # torch.Size([8, 64, 56, 56])
        out = self.width_block(out)                 # torch.Size([8, 64, 56, 56])
        out = self.relu(out)

        out = self.conv_up(out)                 # torch.Size([8, 128, 56, 56])
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, in_channels, layers, span, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = in_channels         # 24
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups   #8
        self.base_width = width_per_group     #64
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, in_channels, layers, kernel_size=span)    # in_channels = 24, layers = 1, span=55
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size, stride=1, dilate=False):     # planes = 24, blocks = 1, kernel_size = 55
        norm_layer = self._norm_layer     # BatchNorm2d
        downsample = None
        previous_dilation = self.dilation      # dilation=1
        if dilate:                              # dilate = False
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,  kernel_size, stride, downsample, groups=self.groups,       #self.inplanes = 24, planes = 24, kernel_size = 55, stride = 1, downsample = None, groups = 6
                            base_width=self.base_width, dilation=previous_dilation,                             # base_width = 64, dilation = 1, norm_layer=BatchNorm2d
                            norm_layer=norm_layer))

        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2
        counter=1
        for _ in range(1, blocks):
            # print('creating blocks : ', counter)
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))
            counter+=1
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # print("before layer 1",x.shape)     # torch.Size([8, 32, 56, 56])
        x = self.layer1(x)
        # print("after layer 1",x.shape)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def axial55s_128():
    model = AxialAttentionNet(AxialBlock, in_channels=128, layers=1, span=250, groups=8)
    return model

def axial55s_128_3layers():
    model = AxialAttentionNet(AxialBlock, in_channels=128, layers=3, span=250, groups=8)
    return model

def axial55s_256():
    model = AxialAttentionNet(AxialBlock, in_channels=256, layers=1, span=55, groups=8)
    return model

def axial55s_256_3layers():
    model = AxialAttentionNet(AxialBlock, in_channels=256, layers=3, span=55, groups=8)
    return model

def axial32s():
    model = AxialAttentionNet(AxialBlock, in_channels = 128, layers=1, span=1238, groups=8)
    return model

def axial_3layers(dim):
    model = AxialAttentionNet(AxialBlock, in_channels = 128, layers=3, span=dim, groups=8)
    return model

def axial_1layer(dim):
    model = AxialAttentionNet(AxialBlock, in_channels=256, layers=1, span=dim, groups=8)
    return model



if __name__=="__main__":
    # model = AxialAttentionNet(AxialBlock, in_channels=24, layers = 1, span = 55, groups = 6)
    model = axial32s_3layers()
    input = torch.rand((10, 128, 32, 32))
    print(model)
    output = model(input)
    # # model_try = AxialBlock(32, 32)
    # # print(model_try)
    # # output_try = model_try(input)
    # # print(output_try.shape)
    print(output.shape)

    # input = torch.randn(10, 24, 55, 55)
    # test_AxialAttention()
    # test_AxialBlock()
