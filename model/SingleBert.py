import torch.nn as nn
import torch
from torch.nn.init import normal_



class PosEmb(nn.Module):
    def __init__(self, hidden_size,initializer_range=0.02) -> None:
        super(PosEmb,self).__init__()
        self.hidden_size = hidden_size
        self._reset_parameters(initializer_range)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)
