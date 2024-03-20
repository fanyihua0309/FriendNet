import torch
from torch import nn


def compute_psnr(pred, GT):
    criterion = nn.MSELoss().cuda()
    psnr = 10 * torch.log10((1.0 / criterion(pred, GT)))
    return psnr
