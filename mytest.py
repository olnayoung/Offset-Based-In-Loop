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
    recon_numpy = np.round(recon_numpy[0,0,:,:])
    recon_numpy = np.array(recon_numpy, dtype=np.int32)

    temp_offset = offset.cpu()
    offset_numpy = np.array(temp_offset.data.numpy())
    offset_numpy = np.round(offset_numpy[0,:,0,0])
    offset_numpy = np.array(offset_numpy, dtype=np.int32)

    recon_numpy = recon_numpy.tolist()
    offset_numpy = offset_numpy.tolist()
    return offset_numpy, recon_numpy

# a = np.random.rand(64*64)*255
# b = np.random.rand(64*64)*255

# offset, recon = test(a, b, 64, 64)

# print(len(recon))
# print(len(recon[0]))