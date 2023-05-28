import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

import numpy as np
import math

EPSILON = 1e-6

class SparsityMetrics(Metric):
    
    is_differentiable: False
    higher_is_better: True
    full_state_update: False
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.is_updated = False
        self.add_state("sparsity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, sparsity):
        self.is_updated = True
        self.sparsity += sparsity.to(self.sparsity.device)
        self.num_examples += 1
    
    def reset(self) -> None:
        self.is_updated = False
        return super().reset()
                
    def compute(self):
        if self.is_updated:
            tens =  self.sparsity / self.num_examples
        else:
            tens = torch.tensor(0.0)
        return tens.item()
    
class SmoothStep(nn.Module):
    """A smooth-step function.
    For a scalar x, the smooth-step function is defined as follows:
    0                                             if x <= -gamma/2
    1                                             if x >= gamma/2
    3*x/(2*gamma) -2*x*x*x/(gamma**3) + 0.5       o.w.
    See https://arxiv.org/abs/2002.07772 for more details on this function.
    """

    def __init__(self, gamma=1.0):
        """Initializes the layer.
        Args:
          gamma: Scaling parameter controlling the width of the polynomial region.
        """
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2
        self._upper_bound = gamma / 2
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def forward(self, x):
        return torch.where(
            x <= self._lower_bound,
            torch.zeros_like(x),
            torch.where(
                x >= self._upper_bound,
                torch.ones_like(x),
                self._a3 * (x**3) + self._a1 * x + self._a0,
            ),
        )
