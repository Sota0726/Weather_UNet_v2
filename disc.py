import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from nets import sn_double_conv
from nets import Block
from nets import OptimizedBlock


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
