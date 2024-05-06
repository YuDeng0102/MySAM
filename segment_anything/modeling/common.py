# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    def __init__(self,
                 in_dim, embeding_feature=64):
        super().__init__()

        self.project1 = nn.Linear(in_dim, embeding_feature)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(embeding_feature, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv2d(embeding_feature, embeding_feature, kernel_size=3, padding=3 // 2,
                               groups=embeding_feature)
        self.conv2 = nn.Conv2d(embeding_feature, embeding_feature, kernel_size=3, padding=3 // 2,
                               groups=embeding_feature)
        self.conv3 = nn.Conv2d(embeding_feature, embeding_feature, kernel_size=3, padding=3 // 2,
                               groups=embeding_feature)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.belta = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.belta
        project1 = self.project1(x)
        project1 = project1.permute(0, 3, 1, 2)

        identity2 = project1
        conv1_x = self.conv1(project1)
        conv2_x = self.conv2(project1)
        conv3_x = self.conv3(project1)
        project1 = (conv1_x + conv2_x + conv3_x) / 3.0 + identity2

        project1 = project1.permute(0, 2, 3, 1)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        return identity + project2