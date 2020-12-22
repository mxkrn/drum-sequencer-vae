import torch
import torch.nn.functional as F


def reconstruction_loss(input, target, channels):
    onsets, velocities, offsets = torch.split(input, channels, dim=-1)
    #target = torch.transpose(target, 0, 1).float()
    target_onsets, target_velocities, target_offsets = torch.split(
        target, channels, dim=-1
    )
    onsets_loss = F.binary_cross_entropy(onsets, target_onsets, reduction="mean")
    velocities_loss = F.mse_loss(velocities, target_velocities, reduction="mean")
    offsets_loss = F.mse_loss(offsets, target_offsets, reduction="mean")
    return onsets_loss + velocities_loss + offsets_loss
