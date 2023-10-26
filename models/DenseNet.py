import torch
import math
import functools
import torch.nn.functional as F
from models.NEDB_IN import Dense_Block_IN,Dense_Block_IN
from torch import nn
from torch.nn import Parameter
import numpy as np
# from models.base_network import BaseNetwork
# from .blocks import ConvBlock, DeconvBlock, MeanShift
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class SEG(nn.Module):
	def __init__(self):
		super(SEG, self).__init__()

		self.up_concat4 = unetUp(128, 64)
		# 128,128,256
		self.up_concat3 = unetUp(128, 64)
		# 256,256,128
		self.up_concat2 = unetUp(128, 64)
		# 512,512,64
		self.up_concat1 = unetUp(128, 64)

		self.final = nn.Conv2d(64, 8, 1)

	def forward(self, x):   # 1, 3, 256, 256   300
		#
		[latent, down_41, down_31, down_21, down_11, feature_neg] = x


		up4 = self.up_concat4(down_41, latent)
		up3 = self.up_concat3(down_31, up4)
		up2 = self.up_concat2(down_21, up3)
		up1 = self.up_concat1(down_11, up2)

		final = self.final(up1)

		return final

		return out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, input_dim, output_dim):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
			nn.Conv2d(input_dim, output_dim, 3, 1, 1),
			nn.InstanceNorm2d(output_dim),
			nn.ReLU(True),
			nn.Conv2d(output_dim, output_dim, 3, 1, 1),
			nn.InstanceNorm2d(output_dim),
			nn.ReLU(True),
		)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class PFC(nn.Module):
	def __init__(self):
		super(PFC, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.Tanh(),
		)

		self.conv_2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(128, affine=True),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(256, affine=True),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(256, affine=True),
			nn.ReLU(),
        )

	def forward(self, x):   # 1, 3, 256, 256   300
		#
		out = self.conv1(x)   # 1, 64, 256, 256    300
		out = self.conv_2(out)

		return out

class Trans_Up(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Up, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode='bilinear')
		out = self.relu(self.IN1(self.conv0(x)))
		return out

# class Trans_Up(nn.Module):
# 	def __init__(self, in_planes, out_planes):
# 		super(Trans_Up, self).__init__()
# 		self.conv0 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1)
# 		self.IN1 = nn.InstanceNorm2d(out_planes)
# 		self.relu = nn.ReLU(inplace=True)
#
# 	def forward(self, x):
# 		out = self.relu(self.IN1(self.conv0(x)))
# 		return out

class Trans_Down(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Down, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out

class Encoder_U(nn.Module):
	def __init__(self):
		super(Encoder_U, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.Tanh(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.Tanh(),
		)
		#
		self.Dense_Down_4 = Dense_Block_IN(block_num=2, inter_channel=32, channel=64)     # 16
		self.Dense_Down_3 = Dense_Block_IN(block_num=2, inter_channel=32, channel=64)     # 32
		self.Dense_Down_2 = Dense_Block_IN(block_num=2, inter_channel=32, channel=64)     # 64
		self.Dense_Down_1 = Dense_Block_IN(block_num=2, inter_channel=32, channel=64)     # 128
		# 下采样
		self.trans_down_1 = Trans_Down(64, 64)
		self.trans_down_2 = Trans_Down(64, 64)
		self.trans_down_3 = Trans_Down(64, 64)
		self.trans_down_4 = Trans_Down(64, 64)

		self.conv_ne = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(512),
			nn.ReLU(),
        )

	def forward(self, x):   # 1, 3, 256, 256   300
		#
		feature_neg = self.conv1(x)   # 1, 64, 256, 256    300
		feature_0 = self.conv2(feature_neg)   # # 1, 64, 256, 256   300
		#######################################################
		down_11 = self.Dense_Down_1(feature_0)  # 1, 64, 256, 256   300
		down_1 = self.trans_down_1(down_11)   # 1, 64, 128, 128     150

		down_21 = self.Dense_Down_2(down_1)  # 1, 64, 128, 128      150
		down_2 = self.trans_down_2(down_21)    # 1, 64, 64, 64      75lf

		down_31 = self.Dense_Down_3(down_2)  # 1, 64, 64, 64        75
		down_3 = self.trans_down_3(down_31)    # 1, 64, 32, 32      38

		down_41 = self.Dense_Down_4(down_3)  # 1, 64, 32, 32        38
		down_4 = self.trans_down_4(down_41)    # 1, 64, 1616      19

		# negetive = self.conv_ne(down_41)

		return [down_4,down_3,down_2,down_1]



class Encoder_A(nn.Module):
	def __init__(self):
		super(Encoder_A, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 16 -> 8
		# 几个 conv, 中间 channel, 输入 channel
		#
		self.Dense_Down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 16
		self.Dense_Down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 32
		self.Dense_Down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 64
		self.Dense_Down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 128
		# 下采样
		self.trans_down_1 = Trans_Down(64, 64)
		self.trans_down_2 = Trans_Down(64, 64)
		self.trans_down_3 = Trans_Down(64, 64)
		self.trans_down_4 = Trans_Down(64, 64)

		self.conv_po = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(128),
            nn.ReLU(),
			nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(512),
			nn.ReLU(),
        )

	def forward(self, x):   # 1, 3, 256, 256   300
		#
		feature_neg = self.conv1(x)   # 1, 64, 256, 256    300
		feature_0 = self.conv2(feature_neg)   # # 1, 64, 256, 256   300
		#######################################################
		down_11 = self.Dense_Down_1(feature_0)  # 1, 64, 256, 256   300
		down_1 = self.trans_down_1(down_11)   # 1, 64, 128, 128     150

		down_21 = self.Dense_Down_2(down_1)  # 1, 64, 128, 128      150
		down_2 = self.trans_down_2(down_21)    # 1, 64, 64, 64      75

		down_31 = self.Dense_Down_3(down_2)  # 1, 64, 64, 64        75
		down_3 = self.trans_down_3(down_31)    # 1, 64, 32, 32      38

		down_41 = self.Dense_Down_4(down_3)  # 1, 64, 32, 32        38
		down_4 = self.trans_down_4(down_41)    # 1, 64, 16, 16      19

		latent = self.Latent(down_4)
		# positive = self.conv_po(down_41)
		#######################################################
		return [latent, down_41, down_31, down_21, down_11,feature_neg]

class Generator_A(nn.Module):
	def __init__(self):
		super(Generator_A, self).__init__()
		#
		self.Dense_Up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.Dense_Up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.Dense_Up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		self.Dense_Up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 32 -> 16
		# 上采样
		self.trans_up_4 = Trans_Up(64, 64)
		# self.trans_up_3 = Trans_Up_1(64, 64)
		self.trans_up_3 = Trans_Up(64, 64)
		self.trans_up_2 = Trans_Up(64, 64)
		self.trans_up_1 = Trans_Up(64, 64)
		# 融合
		self.up_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, 3, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x,y):   # 1, 3, 256, 256   300
		[latent, down_41, down_31, down_21, down_11,feature_neg] = x
		[down_4, down_3, down_2, down_1]  =y

		up_4 = self.trans_up_4(latent+down_4)  # 1, 64, 32, 32             38
		up_4 = torch.cat([down_41, up_4], dim=1)  # 1, 128, 32, 32  38
		up_4 = self.up_4_fusion(up_4)     # 1, 64, 32, 32           38
		up_4 = self.Dense_Up_4(up_4)       # 1, 64, 32, 32          38

		up_3 = self.trans_up_3(up_4+down_3)  # 1, 64, 64, 64               76
		up_3 = torch.cat([down_31, up_3], dim=1)  # 1, 128, 64, 64
		up_3 = self.up_3_fusion(up_3)     # 1, 64, 64, 64
		up_3 = self.Dense_Up_3(up_3)       # 1, 64, 64, 64

		up_2 = self.trans_up_2(up_3+down_2)  # 1, 64, 128,128
		up_2 = torch.cat([up_2, down_21], dim=1)  # 1, 128, 128,128
		up_2 = self.up_2_fusion(up_2)     # 1, 64, 128,128
		up_2 = self.Dense_Up_2(up_2)       # 1, 64, 128,128

		up_1 = self.trans_up_1(up_2+down_1)  # 1, 64, 256, 256
		up_1 = torch.cat([up_1, down_11], dim=1)  # 1, 128, 256, 256
		up_1 = self.up_1_fusion(up_1)     # 1, 64, 256, 256
		up_1 = self.Dense_Up_1(up_1)       # 1, 64, 256, 256
		#
		feature = self.fusion(up_1)  # 1, 64, 256, 256
		#
		feature = feature  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs

class Generator_B(nn.Module):
	def __init__(self):
		super(Generator_B, self).__init__()
		#
		self.conv1 = nn.Sequential(
			nn.Conv2d(128, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.Dense_Up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.Dense_Up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.Dense_Up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		self.Dense_Up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 32 -> 16
		# 上采样
		self.trans_up_4 = Trans_Up(64, 64)
		# self.trans_up_3 = Trans_Up_1(64, 64)
		self.trans_up_3 = Trans_Up(64, 64)
		self.trans_up_2 = Trans_Up(64, 64)
		self.trans_up_1 = Trans_Up(64, 64)
		# 融合
		self.up_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, 3, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x):   # 1, 3, 256, 256   300
		[Latent, down_41, down_31, down_21, down_11,feature_neg] = x

		#######################################################
		#######################################################

		up_4 = self.trans_up_4(Latent)  # 1, 64, 32, 32             38
		up_4 = torch.cat([down_41, up_4], dim=1)  # 1, 128, 32, 32  38
		up_4 = self.up_4_fusion(up_4)     # 1, 64, 32, 32           38
		up_4 = self.Dense_Up_4(up_4)       # 1, 64, 32, 32          38

		up_3 = self.trans_up_3(up_4)  # 1, 64, 64, 64               76
		up_3 = torch.cat([down_31, up_3], dim=1)  # 1, 128, 64, 64
		up_3 = self.up_3_fusion(up_3)     # 1, 64, 64, 64
		up_3 = self.Dense_Up_3(up_3)       # 1, 64, 64, 64

		up_2 = self.trans_up_2(up_3)  # 1, 64, 128,128
		up_2 = torch.cat([up_2, down_21], dim=1)  # 1, 128, 128,128
		up_2 = self.up_2_fusion(up_2)     # 1, 64, 128,128
		up_2 = self.Dense_Up_2(up_2)       # 1, 64, 128,128

		up_1 = self.trans_up_1(up_2)  # 1, 64, 256, 256
		up_1 = torch.cat([up_1, down_11], dim=1)  # 1, 128, 256, 256
		up_1 = self.up_1_fusion(up_1)     # 1, 64, 256, 256
		up_1 = self.Dense_Up_1(up_1)       # 1, 64, 256, 256
		#
		feature = self.fusion(up_1)  # 1, 64, 256, 256
		#
		feature = feature  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs


class NLEDN_IN_32_16_32(nn.Module):
	def __init__(self, input_nc, output_n):
		super(NLEDN_IN_32_16_32, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		# 几个 conv, 中间 channel, 输入 channel
		self.Dense_Up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.Dense_Up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.Dense_Up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		self.Dense_Up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 32 -> 16
		#
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 16 -> 8
		#
		self.Dense_Down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 16
		self.Dense_Down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 32
		self.Dense_Down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 64
		self.Dense_Down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 128
		# 下采样
		self.trans_down_1 = Trans_Down(64, 64)
		self.trans_down_2 = Trans_Down(64, 64)
		self.trans_down_3 = Trans_Down(64, 64)
		self.trans_down_4 = Trans_Down(64, 64)
		# 上采样
		self.trans_up_4 = Trans_Up(64, 64)
		# self.trans_up_3 = Trans_Up_1(64, 64)
		self.trans_up_3 = Trans_Up(64, 64)
		self.trans_up_2 = Trans_Up(64, 64)
		self.trans_up_1 = Trans_Up(64, 64)
		# 融合
		self.up_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.up_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, output_n, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x):   # 1, 3, 256, 256   300
		#
		feature_neg = self.conv1(x)   # 1, 64, 256, 256    300
		feature_0 = self.conv2(feature_neg)   # # 1, 64, 256, 256   300
		#######################################################
		down_11 = self.Dense_Down_1(feature_0)  # 1, 64, 256, 256   300
		down_1 = self.trans_down_1(down_11)   # 1, 64, 128, 128     150

		down_21 = self.Dense_Down_2(down_1)  # 1, 64, 128, 128      150
		down_2 = self.trans_down_2(down_21)    # 1, 64, 64, 64      75

		down_31 = self.Dense_Down_3(down_2)  # 1, 64, 64, 64        75
		down_3 = self.trans_down_3(down_31)    # 1, 64, 32, 32      38

		down_41 = self.Dense_Down_4(down_3)  # 1, 64, 32, 32        38
		down_4 = self.trans_down_4(down_41)    # 1, 64, 16, 16      19

		#######################################################
		Latent = self.Latent(down_4)  # 1, 64, 16, 16               19
		#######################################################

		up_4 = self.trans_up_4(Latent)  # 1, 64, 32, 32             38
		up_4 = torch.cat([down_41, up_4], dim=1)  # 1, 128, 32, 32  38
		up_4 = self.up_4_fusion(up_4)     # 1, 64, 32, 32           38
		up_4 = self.Dense_Up_4(up_4)       # 1, 64, 32, 32          38

		up_3 = self.trans_up_3(up_4)  # 1, 64, 64, 64               76
		up_3 = torch.cat([down_31, up_3], dim=1)  # 1, 128, 64, 64
		up_3 = self.up_3_fusion(up_3)     # 1, 64, 64, 64
		up_3 = self.Dense_Up_3(up_3)       # 1, 64, 64, 64

		up_2 = self.trans_up_2(up_3)  # 1, 64, 128,128
		up_2 = torch.cat([up_2, down_21], dim=1)  # 1, 128, 128,128
		up_2 = self.up_2_fusion(up_2)     # 1, 64, 128,128
		up_2 = self.Dense_Up_2(up_2)       # 1, 64, 128,128

		up_1 = self.trans_up_1(up_2)  # 1, 64, 256, 256
		up_1 = torch.cat([up_1, down_11], dim=1)  # 1, 128, 256, 256
		up_1 = self.up_1_fusion(up_1)     # 1, 64, 256, 256
		up_1 = self.Dense_Up_1(up_1)       # 1, 64, 256, 256
		#
		feature = self.fusion(up_1)  # 1, 64, 256, 256
		#
		feature = feature + feature_neg  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs