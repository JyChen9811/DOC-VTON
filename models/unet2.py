import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv
class UNet(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=14):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
class UNet1(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=3):
        super(UNet1, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=3):
        super(UNet2, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class MaskWeightedCrossEntropyLoss(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super(MaskWeightedCrossEntropyLoss, self).__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        mask = mask.byte()
        target_inmask = target[mask]
        target_outmask = target[~mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()

        predict_inmask = predict[mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        predict_outmask = predict[(~mask).view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss_inmask = nn.functional.cross_entropy(
            predict_inmask, target_inmask, size_average=False)
        loss_outmask = nn.functional.cross_entropy(
            predict_outmask, target_outmask, size_average=False)
        loss = (self.inmask_weight * loss_inmask + self.outmask_weight * loss_outmask) / (n * h * w)
        return loss