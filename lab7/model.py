import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# def weights_init(self):
#     for m in self._modules:
#         if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
#             self._modules[m].weight.data.normal_(0, 0.02)
#             self._modules[m].bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, img_shape, c_dim):
        super(Discriminator, self).__init__()
        self.H, self.W, self.C = img_shape
        self.conditionExpand = nn.Sequential(
            nn.Linear(24, self.H*self.W*1),
            # nn.Linear(24,self.C*self.H*self.W),
            # nn.LeakyReLU()
        )

        ndf = 64

        self.conv1 = nn.Sequential(
                nn.Conv2d(4, ndf, (4, 4), stride=(2,2), padding=(1,1), bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(ndf, ndf*2, (4, 4), stride=(2,2), padding=(1,1), bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(ndf*2, ndf*4, (4, 4), stride=(2,2), padding=(1,1), bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
                nn.Conv2d(ndf*4, ndf*8, (4, 4), stride=(2,2), padding=(1,1), bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU()
        )
        self.conv5 = nn.Conv2d(ndf*8, 1, (4, 4), stride=(1,1))
        self.sigmoid = nn.Sigmoid()


    def forward(self,X,c):
        """
        :param X: (batch_size,3,64,64) tensor
        :param c: (batch_size,24) tensor
        :return: (batch_size) tensor
        """
        c = self.conditionExpand(c).view(-1, 1, self.H, self.W)
        # c=self.conditionExpand(c).view(-1,self.C,self.H,self.W)
        out = torch.cat((X,c), dim=1)  # become(N,4,64,64)
        out = self.conv1(out)  # become(N,64,32,32)
        out = self.conv2(out)  # become(N,128,16,16)
        out = self.conv3(out)  # become(N,256,8,8)
        out = self.conv4(out)  # become(N,512,4,4)
        out = self.conv5(out)  # become(N,1,1,1)
        out = self.sigmoid(out)  # output value between [0,1]
        out = out.view(-1)
        return out



class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.conditionExpand = nn.Sequential(
            nn.Linear(24, c_dim),
            # nn.ReLU(True)
        )

        ngf = 64

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim+c_dim, ngf*8, (4, 4), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, (4, 4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, (4, 4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, (4, 4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        self.convT5 = nn.ConvTranspose2d(ngf, 3, (4, 4), stride=(2,2), padding=(1,1), bias=False)
        self.tanh = nn.Tanh()



    def forward(self,z,c):
        """
        :param z: (batch_size,100) tensor
        :param c: (batch_size,24) tensor
        :return: (batch_size,3,64,64) tensor
        """
        z = z.view(-1, self.z_dim, 1, 1)
        c = self.conditionExpand(c).view(-1, self.c_dim, 1, 1)
        out = torch.cat((z,c), dim=1)  # become(N,z_dim+c_dim,1,1)
        out = self.convT1(out)  # become(N,512,4,4)
        out = self.convT2(out)  # become(N,256,8,8)
        out = self.convT3(out)  # become(N,128,16,16)
        out = self.convT4(out)  # become(N,64,32,32)
        out = self.convT5(out)  # become(N,3,64,64)
        out = self.tanh(out)    # output value between [-1,+1]
        return out
