import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


def loss_fn(recon_x, x, mu, logvar, beta):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # CE = F.cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = - 0.5* torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * beta

    return MSE + KLD, MSE, KLD
