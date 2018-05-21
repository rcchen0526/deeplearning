import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from logger import Logger
import numpy as np
import sys

class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()
    
    self.main = nn.Sequential(
      nn.Conv2d(1024, 1, 1),
      nn.Sigmoid()
    )
    

  def forward(self, x):
    output = self.main(x).view(-1, 1)
    return output


class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()

    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, 10, 1)
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)

  def forward(self, x):

    y = self.conv(x)

    disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var 


class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x)
    return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

d = {'2' : 0,'5' : 1,  '3' : 2, '1' : 3, '4' : 4, '9' : 5, '7' : 6, '6' : 7, '8' : 8, '0' : 9}
dis_c = torch.FloatTensor(50, 10).cuda()
con_c = torch.FloatTensor(50, 2).cuda()
noise = torch.FloatTensor(50, 62).cuda()
fix_noise = torch.Tensor(50, 62).uniform_(-1, 1)
noise.data.copy_(fix_noise)

idx = np.arange(10).repeat(5)
one_hot = np.zeros((50, 10))
one_hot[range(50), idx] = 1

c = np.linspace(-1, 1, 50).reshape(1, -1)
c = np.repeat(c, 1, 0).reshape(-1, 1)
c1 = np.hstack([c, np.zeros_like(c)])

dis_c.data.copy_(torch.Tensor(one_hot))

con_c.data.copy_(torch.from_numpy(c1))
z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
g = G()
print('Generating...')
g = torch.load('model_infogan_g')
g.eval()
x_save = g(z)
demo = torch.zeros(5, 1, 28, 28).cuda()
for i in range(5):
  demo[i][0] += x_save[5*d[sys.argv[1]]+i][0]
#x_save = x_save[d[sys.argv[1]]]
save_image(demo.data, 'Img_infogan_' + str(sys.argv[1]) + '.png', nrow=10)
print('Finish!')
