import numpy as np
import torch
from torch import nn
import os.path as osp
from torch.nn import functional as F


def np_softmax(x, axis=0):
    dense = np.exp(x)
    dense = dense / (np.sum(dense, axis=axis)+0.00001)
    return dense

class SuperPointNet(nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn. MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 1 x H x W.
        Output
        semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc

class SuperPointFrontend():
    def __init__(self, project_root):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.cell = 8  # Size of each output cell. Keep this fixed.
        try:
            checkpoint = osp.join(project_root,'checkpoints','superpoint', 'superpoint_v1.pth')
        except:
            checkpoint = 'checkpoints/pretrained_pth/superpoint_v1.pth'
        print(project_root, 'Load SuperPoint Pth from ', checkpoint)
        self.net = SuperPointNet()
        self.net.load_state_dict(torch.load(checkpoint))
        self.net.eval()
        self.net = self.net.to(self.device)
        # self.net = self.net

    def run(self, input):
        # for superpoint the input should be (480,752,1), gray_scale
        # input: torch.tensor(8,1,480,752)
        input = input.type(torch.FloatTensor) # (8,1,480,752)
        outputs = self.net(input.to(self.device))
        #outputs = self.net(input)
        semi, coarse_desc = outputs[0], outputs[1] # (8,65,60,94),(8,128,60,94)
        dense = F.softmax(semi, dim=1)
        dense = dense.detach().cpu().numpy()
        # dense = np_softmax(semi, axis=1)
        coarse_desc=coarse_desc.detach().cpu().numpy()
        ret = {
            'dense_scores': dense,
            'local_descriptor_map': coarse_desc
        }
        return ret


