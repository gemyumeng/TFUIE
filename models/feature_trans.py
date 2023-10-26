import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ssd_test.nets.mobilenetv2 import InvertedResidual, mobilenet_v2

from ssd_test.nets.vgg import vgg as add_vgg

class FT(nn.Module):
    def __init__(self):
        super(FT, self).__init__()
        self.conv_9 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv_16 = nn.Conv2d(256, 64, 1, 1, 0)
        self.conv_23 = nn.Conv2d(512, 64, 1, 1, 0)

    def forward(self, x_9, x_16, x_23):
        f_9 = self.conv_9(x_9)
        f_16 = self.conv_16(x_16)
        f_23 = self.conv_23(x_23)
        return f_9, f_16, f_23