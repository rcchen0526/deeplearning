import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import math
from torch.autograd import Variable

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class CNN(nn.Module):
	def __init__(self, block, layers):
		self.in_channels = 16
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(	# input shape (3, 32, 32)
				nn.Conv2d(
				in_channels=3,		# input height
				out_channels=16,	# n_filters
				kernel_size=3,		# filter size
				stride=1,		# filter movement/step
				padding=1,		#
				bias=False),				# output shape (64, 32, 32)
			#nn.ReLU(),		# activation
			#nn.MaxPool2d(kernel_size=2),	# output shape (64, 32, 32)
		)
		self.bn1 = nn.BatchNorm2d(16)

		#self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 16, layers[0])
		self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

		self.linear = nn.Linear(64 * 4 * 4, 10)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.in_channels:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channels, planes * block.expansion,
					kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.in_channels, planes, stride, downsample))
		self.in_channels = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_channels, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = F.avg_pool2d(x, 2)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		return x

def test():
	print('Testing...')
	cnn.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		outputs = cnn(inputs)

		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

	print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))


transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2611)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

cnn = torch.load(sys.argv[1])
cnn.cuda()
cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
def main():
	test()	
	

if __name__ == "__main__":
	main()

