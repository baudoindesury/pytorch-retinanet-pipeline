import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import numpy as np


class SuperPointNet_Detector(torch.nn.Module):
    def __init__(self, input_channel):
        super(SuperPointNet_Detector, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 128, 256
        det_h = 65
        # Detector Head.
        self.convP0 = torch.nn.Conv2d(input_channel, c4, kernel_size=3, stride=1, padding=1)
        self.bnP0 = nn.BatchNorm2d(c4)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        self.relu = torch.nn.ReLU(inplace=True)
        list = []

        list.append(self.convP0)
        list.append(self.bnP0)
        list.append(self.relu)
        list.append(self.convPa)
        list.append(self.bnPa)
        list.append(self.relu)
        list.append(self.convPb)
        list.append(self.bnPb)

        self.detector = nn.Sequential(*list)
    def forward(self,x):
        # cP0 = self.relu(self.bnP0(self.convP0(x)))
        # cPa = self.relu(self.bnPa(self.convPa(cP0)))
        # semi = self.bnPb(self.convPb(cPa))
        semi = self.detector(x)
        return semi

if __name__ == '__main__':
    model = SuperPointNet_Detector(input_channel=40)
    inputs = torch.randn(1,40,60,94)
    outputs = model(inputs)
    print(model)
    print('Inputs shape:', inputs.shape)
    print('Outputs shape:', outputs.shape)
