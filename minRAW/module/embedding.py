import torch
import torch.nn as nn
from typing import Dict

class GPSEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        return self.layers(x)

