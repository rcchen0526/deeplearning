import matplotlib.pyplot as plt
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
#import argparse
from torch.autograd import Variable
from logger import Logger
import tensorflow as tf

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
def resnet20():
	return CNN(BasicBlock, [3,3,3])

def resnet56():
	return CNN(BasicBlock, [9,9,9])

def resnet110():
	return CNN(BasicBlock, [18,18,18])

def train(epoch):
	global train_iter
	global logger
	print(epoch)
	cnn.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		label = targets
		inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = cnn(inputs)
		loss = loss_function(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		info = {'Train_loss' : loss.data[0], 'Train_Error%' : 100 - 100 * correct/total}
		for tag, value in info.items():
			logger.scalar_summary(tag, value, train_iter)
		train_iter += 1
	print('Train : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
	global best_acc
	global test_iter
	global logger
	cnn.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		outputs = cnn(inputs)
		loss = loss_function(outputs, targets)

		test_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		#test_error.append(round(correct/total, 3))
		info = {'Test_Loss' : loss.data[0], 'Test_Error%' : 100 - 100 * correct/total}
		for tag, value in info.items():
			logger.scalar_summary(tag, value, test_iter)
		test_iter += 1
	if best_acc > correct/total:
		best_acc = correct/total
		torch.save(cnn, './model20')

	print('Test : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
best_acc = 0
train_iter = 0
test_iter = 0
LR = 0.1
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2611)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2611)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

cnn = resnet20()
cnn.cuda()
cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
logger = Logger('./logs')
def main():
	global LR
	for epoch in range(1, 165):
		if epoch == 81 or epoch == 122:
			LR /= 10
		train(epoch)
		test(epoch)	
	

if __name__ == "__main__":
	main()

