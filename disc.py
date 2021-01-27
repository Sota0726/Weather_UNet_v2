import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils
from functools import partial
import math

from nets import sn_double_conv
from nets import Block
from nets import OptimizedBlock
from nets import conv3x3x3, downsample_basic_block, BasicBlock3D


# 224 -> conv1(stride=2) or pool1(2) -> 112 -> conv2 -> 56 -> conv3 -> 28 -> conv4 -> 14 -> conv5 -> 7
# 256 -> conv1(stride=2) or pool1(2) -> 128 -> conv2 -> 64 -> conv3 -> 32 -> conv4 -> 16 -> conv5 -> 8 -> conv6 -> 4 -> conv7 -> 2


class SNDisc(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = sn_double_conv(3, 64)
        self.conv2 = sn_double_conv(64, 128)
        self.conv3 = sn_double_conv(128, 256)
        self.conv4 = sn_double_conv(256, 512)
        [nn.init.xavier_uniform_(
            getattr(self, 'conv{}'.format(i))[j].weight,
            np.sqrt(2)
            ) for i in range(1, 5) for j in range(2)]

        self.l = nn.utils.spectral_norm(nn.Linear(512, 1))
        nn.init.xavier_uniform_(self.l.weight)

        self.embed = nn.utils.spectral_norm(nn.Linear(num_classes, 512, bias=True))
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x, c=None):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = torch.sum(c4, [2, 3])  # global pool
        out = self.l(x)
        e_c = self.embed(c)
        if c is not None:
            out += torch.sum(e_c * x, dim=1, keepdim=True)
        # out = nn.Sigmoid(out)
        return [out, c1, c2, c3, c4]


class SNDisc_(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = sn_double_conv(3, 64)
        self.conv2 = sn_double_conv(64, 128)
        self.conv3 = sn_double_conv(128, 256)
        self.conv4 = sn_double_conv(256, 512)
        self.conv5 = sn_double_conv(512, 1024)
        [nn.init.xavier_uniform_(
            getattr(self, 'conv{}'.format(i))[j].weight,
            np.sqrt(2)
            ) for i in range(1, 5) for j in range(2)]

        self.l = nn.utils.spectral_norm(nn.Linear(1024, 1))
        nn.init.xavier_uniform_(self.l.weight)

        self.embed = nn.utils.spectral_norm(nn.Linear(num_classes, 1024, bias=True))
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x, c=None):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        x = torch.sum(c5, [2, 3])  # global pool
        out = self.l(x)
        e_c = self.embed(c)
        if c is not None:
            out += torch.sum(e_c * x, dim=1, keepdim=True)
        # out = nn.Sigmoid(out)
        # return [out, c1, c2, c3, c4, c5]
        return [out]


# refarence code https://github.com/crcrpar/pytorch.sngan_projection 
class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_classes, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = 64
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, self.num_features)
        self.block2 = Block(self.num_features, self.num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(self.num_features * 2, self.num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(self.num_features * 4, self.num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(self.num_features * 8, self.num_features * 16,
                            activation=activation, downsample=True)
        self.block6 = Block(self.num_features * 16, self.num_features * 16,
                            activation=activation, downsample=True)
        self.l7 = utils.spectral_norm(nn.Linear(self.num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                # nn.Embedding(num_classes, self.num_features * 16))
                nn.Linear(num_classes, self.num_features * 16, bias=True))
        self._initialize()

    def _initialize(self):
        # --- original --- #
        # init.xavier_uniform_(self.l7.weight.data)
        # optional_l_y = getattr(self, 'l_y', None)
        # if optional_l_y is not None:
        #     init.xavier_uniform_(optional_l_y.weight.data)
        init.xavier_uniform_(self.l7.weight)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return [output]


class SNResNet64ProjectionDiscriminator(nn.Module):

    def __init__(self, num_classes, activation=F.relu):
        super(SNResNet64ProjectionDiscriminator, self).__init__()
        self.num_features = 64
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, self.num_features)
        self.block2 = Block(self.num_features, self.num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(self.num_features * 2, self.num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(self.num_features * 4, self.num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(self.num_features * 8, self.num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(self.num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                # nn.Embedding(num_classes, self.num_features * 16))
                nn.Linear(num_classes, self.num_features * 16, bias=True))

        self._initialize()

    def _initialize(self):
        # init.xavier_uniform_(self.l6.weight.data)
        # optional_l_y = getattr(self, 'l_y', None)
        # if optional_l_y is not None:
        #     init.xavier_uniform_(optional_l_y.weight.data)
        init.xavier_uniform_(self.l6.weight)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return [output]


# reference from https://github.com/tomrunia/PyTorchConv3D
class SNresDisc_3DCNN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 spatial_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=1):
        self.inplanes = 64
        super(SNresDisc_3DCNN, self).__init__()
        self.conv1 = utils.spectral_norm(nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False))
        # self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(spatial_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = utils.spectral_norm(nn.Linear(512 * block.expansion, num_classes))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = utils.spectral_norm(nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize(self):
        init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet10_3d(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = SNresDisc_3DCNN(BasicBlock3D, [1, 1, 1, 1], **kwargs)
    return model


def resnet18_3d(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = SNresDisc_3DCNN(BasicBlock3D, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_3d(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = SNresDisc_3DCNN(BasicBlock3D, [3, 4, 6, 3], **kwargs)
    return model
