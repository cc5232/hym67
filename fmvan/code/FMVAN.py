import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

class BSConvU(nn.Module):
    def __init__(
        self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()


        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class FMB(nn.Module):

    def __init__(self,in_channels,out_channels,bias=True,scal=2):
        super().__init__()

        self.remaining_channels = in_channels // scal
        self.other_channels = in_channels - self.remaining_channels
        self.pw1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
            stride=1,padding=0,dilation=1,groups=1,bias=False,
        )

        self.pdw5 = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.remaining_channels,
            kernel_size=5,
            padding=2,
            dilation=1,
            groups=self.remaining_channels,
            bias=bias,
        )
        self.pdw3 = nn.Conv2d(
            in_channels=self.other_channels,
            out_channels=self.other_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=self.other_channels,
            bias=bias,
        )

    def forward(self, fea):
        fea1, fea2 = torch.split(fea, [self.remaining_channels, self.other_channels], dim=1)
        fea1 = self.pdw5(fea1)
        fea2 = self.pdw3(fea2)
        fea = torch.cat([fea1, fea2], dim=1)
        fea = self.pw1(fea)
        return fea


class ApproximateVariancePooling(nn.Module):

    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.ap = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)
        self.mp = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
    def forward(self, x):
        return self.mp((self.ap(x) - x) ** 2)

class VarAttention(nn.Module):

    def __init__(self, channels, reduce = 4):
        super().__init__()
        zip = channels//reduce
        self.body = nn.Sequential(
            ApproximateVariancePooling(7),
            nn.Conv2d(channels, zip, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip, channels, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w

class EnchanceAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cam = nn.Sequential(

            nn.Conv2d(dim,dim, 3, 1, 1,groups=dim),
            nn.GELU(),
            VarAttention(dim),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.cam(x)

class FMIE(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = FMB(in_channels, self.rc)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = FMB(self.rc, self.rc)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = FMB(self.rc, self.rc)
        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1, 1, 0)
        self.atten = EnchanceAttention(in_channels)

        self.pixel_norm = nn.LayerNorm(in_channels)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out = self.atten(out)
        out = out.permute(0, 2, 3, 1)
        out = self.pixel_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out+input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])

@ARCH_REGISTRY.register()
class SRnet(nn.Module):
    def __init__(self,num_feat=56,num_block=10,upscale=4,rgb_mean=(0.4488, 0.4371, 0.4040)):
        super().__init__()
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.fea_conv = nn.Conv2d(3, num_feat, 3,1,1)
        self.B1 = FMIE(in_channels=num_feat)
        self.B2 = FMIE(in_channels=num_feat)
        self.B3 = FMIE(in_channels=num_feat)
        self.B4 = FMIE(in_channels=num_feat)
        self.B5 = FMIE(in_channels=num_feat)
        self.B6 = FMIE(in_channels=num_feat)
        self.B7 = FMIE(in_channels=num_feat)
        self.B8 = FMIE(in_channels=num_feat)
        self.B9 = FMIE(in_channels=num_feat)
        self.B10 = FMIE(in_channels=num_feat)
        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, 1, 0)
        self.GELU = nn.GELU()
        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)
        self.upsampler = UpsampleOneStep(num_feat, 3, upscale_factor=upscale)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean

        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        out_B9 = self.B9(out_B8)
        out_B10 = self.B10(out_B9)

        trunk = torch.cat(
            [out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8,out_B9, out_B10], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)
        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr) + self.mean
        return output


if __name__ == '__main__':
   print()

