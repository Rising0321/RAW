import torch
import torch.nn as nn


class PadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """
    def __init__(self, pad_idx=[-1,-2]):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-2] + (1, offset,),
                              dtype=torch.bool, device=x.device)

        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape[:2] + mask.shape[-2:])


class FutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)

        mask = future.view((1,) * (x.ndim -2) + future.size())
        return mask.expand(x.shape[:2] + mask.shape[-1:])
