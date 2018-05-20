from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from logger import Logger

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num', type=int, default=0)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.cuda.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Sequential(nn.Linear(30, 392), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(2, 11, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(11),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def encode(self, x, c):
        x = x.view(self.batch_size, 1, 28, 28)
        onehot = torch.zeros([self.batch_size, 10, 784], dtype=torch.float)
        for i in range(self.batch_size):
            for j in range(10):
                if j == c[i]:
                    for k in range(784):
                        onehot[i][j][k] = 1.
        onehot = onehot.view(self.batch_size, 10, 28, 28).cuda()
        x = torch.cat((x, onehot), 1)
        x = Variable(x)
        x = self.conv1(x)
        #print(x.shape)
        h1 = self.fc1(x.view(-1, 784))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, c):
        self.batch_size = c.shape[0]
        onehot = torch.zeros([self.batch_size, 10], dtype=torch.float)
        for i in range(self.batch_size):
            for j in range(10):
                if j == c[i]:
                    onehot[i][j] = 1.
        onehot = Variable(onehot)
        z = torch.cat((z, onehot.cuda()), 1)
        h3 = self.fc3(z)
        h3 = h3.view(self.batch_size, 2, 14, 14).cuda()
        h3 = self.conv2(h3)
        return h3.view(-1, 784)

    def forward(self, x, c):
        self.batch_size = c.shape[0]
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


#model = VAE().to(device)
model = VAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def main():
    print('Generating...')
    model = torch.load('model_vae')
    model.eval()
    sample = torch.randn(1, 20).cuda()
    print('The number is :', args.num)
    finish = [args.num]
    sample = model.decode(sample, torch.tensor(finish, ).cuda()).cpu()
    save_image(sample.view(1, 1, 28, 28),
        'Img_cvae_' + str(args.num) + '.png')
    print('Finish!')

if __name__ == "__main__":
    main()
