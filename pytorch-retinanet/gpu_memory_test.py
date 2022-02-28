import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from retinanet.networks.superpoint_pytorch import SuperPointFrontend

import GPUtilext
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

if __name__ == '__main__':
    attrlist = [[
        {'attr': 'id', 'name': 'ID'},
        {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
        {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
         'precision': 0}],
        [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
         {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
         {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}]]
    GPUtilext.showUtilization(attrList=attrlist)
    superpoint = SuperPointFrontend(project_root='')
    GPUtilext.showUtilization(attrList=attrlist)
