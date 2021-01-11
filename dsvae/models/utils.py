import numpy as np
import torch
import torch.nn as nn


class NoteDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(input.shape, dtype=torch.float, device=input.device)
        zeros = torch.zeros(input.shape, dtype=torch.float, device=input.device)
        mask = torch.rand(input.shape, dtype=torch.float, requires_grad=False)
        mask = torch.where(mask <= p.to(input.device), ones, zeros)
        return torch.mul(input, mask)
