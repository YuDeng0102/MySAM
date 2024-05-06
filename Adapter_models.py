import torch.nn as nn
import torch
import numpy as np

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.out_indices = out_indices
        if len(out_indices) == 4:
            self.fc1 = nn.Conv2d(inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
            self.fc2 = nn.Conv2d(2 * inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
            self.fc3 = nn.Conv2d(4 * inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
            self.fc4 = nn.Conv2d(4 * inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.fc2 = nn.Conv2d(2 * inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
            self.fc3 = nn.Conv2d(4 * inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
            self.fc4 = nn.Conv2d(4 * inplanes, embed_dim,
                                 kernel_size=1, stride=1, padding=0, bias=True)
            
    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        if len(self.out_indices) == 4:
            c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        bs, dim, _, _ = c2.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
    

if __name__=='__main__':
    x=torch.randn([1,3,1024,1024])
    SPM=SpatialPriorModule()
    print(SPM(x)[0].shape)
