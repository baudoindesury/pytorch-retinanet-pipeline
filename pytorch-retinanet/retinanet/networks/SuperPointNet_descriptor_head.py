import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import numpy as np


class SuperPointNet_Descriptor(torch.nn.Module):
    def __init__(self, input_channel):
        super(SuperPointNet_Descriptor, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128 , 256, 256
        # Detector Head.
        self.convD0 = torch.nn.Conv2d(input_channel, c4,kernel_size=3, stride=1, padding=1)
        self.bnD0 = nn.BatchNorm2d(c4)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.relu = torch.nn.ReLU(inplace=True)

        # list = []
        #
        # list.append(self.convD0)
        # list.append(self.bnD0)
        # list.append(self.relu)
        # list.append(self.convDa)
        # list.append(self.bnDa)
        # list.append(self.relu)
        # list.append(self.convDb)
        # list.append(self.bnDb)

        # self.descriptor = nn.Sequential(*list)
    def forward(self,x):
        cD0 = self.relu(self.bnD0(self.convD0(x)))
        cDa = self.relu(self.bnDa(self.convDa(cD0)))
        desc = self.bnDb(self.convDb(cDa))

        return desc #(1,256,60,94)

if __name__ == '__main__':
    model = SuperPointNet_Descriptor()
    inputs = torch.randn(1,128,60,94)
    outputs = model(inputs)
    print('Inputs shape:', inputs.shape)
    print('Outputs shape:', outputs.shape)
