from __future__ import print_function
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from torch.autograd import Variable
import os
import math
import copy
import common

def make_model(args, parent=False):
    return Net()

class Net(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(Net, self).__init__()

        pos_head = [conv(1, 128, 7)]
        pos_body = [common.ResBlock(conv, 128, 3, res_scale=1) for _ in range(5)]
        pos_tail = [conv(128, 4, 3), nn.Tanh()]

        self.pos_head = nn.Sequential(*pos_head)
        self.pos_body = nn.Sequential(*pos_body)
        self.pos_tail = nn.Sequential(*pos_tail)

        def block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )


        offset_head = [conv(2, 256, 7)]
        offset_body = [block(256, 128),
                       block(128, 64),
                       block(64, 32),
                       block(32, 16),
                       block(16, 8),
                       block(8, 4)]

        self.offset_head = nn.Sequential(*offset_head)
        self.offset_body = nn.Sequential(*offset_body)


    def forward(self, x, y):

        pos = self.pos_head(x)
        pos = self.pos_body(pos)
        pos = self.pos_tail(pos)

        xy = torch.cat([x, y], dim=1)
        off = self.offset_head(xy)
        off = self.offset_body(off)

        res4 = pos * off.expand_as(pos)
        off_1, off_2, off_3, off_4 = torch.split(res4, 1, 1)

        return x + off_1 + off_2 + off_3 + off_4, off


def test(input, label, wid, hei):
    input_re = np.zeros((int(hei), int(wid)))
    temp = 0
    for j in range(int(hei)):
        for i in range(int(wid)):
            input_re[j][i] = input[temp]
            temp = temp + 1
    input_re = np.float32(input_re)

    label_re = np.zeros((int(hei), int(wid)))
    temp = 0
    for j in range(int(hei)):
        for i in range(int(wid)):
            label_re[j][i] = label[temp]
            temp = temp + 1
    label_re = np.float32(label_re)

    model = Net()
    model = model.cuda()

    model.load_state_dict(torch.load("model/180_epoch_prams.pt"))
    model.eval()

    LR_img_torch = torch.from_numpy(input_re)
    LR_img_torch.unsqueeze_(0)
    LR_img_torch.unsqueeze_(0)
    LR_img_torch = Variable(LR_img_torch.cuda())

    LR_lab_torch = torch.from_numpy(label_re)
    LR_lab_torch.unsqueeze_(0)
    LR_lab_torch.unsqueeze_(0)
    LR_lab_torch = Variable(LR_lab_torch.cuda())

    recon, offset = model(LR_img_torch, LR_lab_torch)
    temp_recon = recon.cpu()
    recon_numpy = np.array(temp_recon.data.numpy())
    recon_numpy = np.array(recon_numpy, dtype=np.int32)

    temp_offset = offset.cpu()
    offset_numpy = np.array(temp_offset.data.numpy())
    offset_numpy = np.array(offset_numpy, dtype=np.int32)

    offset = np.array([1, 0, 0, 0])

    input_re = input_re.tolist()
    offset = offset.tolist()
    return offset, input_re