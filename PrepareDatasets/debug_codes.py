import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        # x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.softmax(x, dim=-1)
        # print('x1:', x.shape)
        x = F.linear(x, self.project.type_as(x))
        # print('x2:', x.shape)
        # x = x.reshape(-1, 1)
        return x

# Create an example vector of length 20
x = np.random.rand(10)
print(x)

# Find the index of the maximum value in x
# max_index = np.argmax(x)
x = torch.tensor(x)
print(Integral(x))
