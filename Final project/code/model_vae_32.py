import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        # print(input.size(3))
        # print(input.size(0))
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1152):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1152, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print('h', h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        # print('x', x.shape)
        z, mu, logvar = self.encode(x)
        # print('z1', z.shape)
        z = self.decode(z)
        # print('z2', z.shape)
        return z, mu, logvar