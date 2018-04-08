#!/usr/bin/env python2.7
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
#from logger import Logger
import tensorflow as tf
import downsampler as D
import numpy as np
import cv2 

downs = []
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample = 0, skips = 0):
		super(BasicBlock, self).__init__()
		self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv_down = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.bn2 = nn.BatchNorm2d(inplanes)
		self.bn_skip = nn.BatchNorm2d(skips)
		self.relu = nn.LeakyReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.skip = nn.Conv2d(inplanes, skips, kernel_size=1, stride=stride, padding=0, bias=False)
		self.conv_skip = nn.Conv2d(inplanes+skips, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.downsample = downsample
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
		self.skips = skips
		
	def forward(self, x):
		global downs
		print(self.downsample, self.skips)
		if self.downsample == 0:
			x = self.conv(x)
			x = self.conv_down(x)
			x = self.bn1(x)
			x = self.relu(x)
			x = self.conv2(x)
			x = self.bn1(x)
			x = self.relu(x)
			downs.append(x)
		elif self.skips != 0 :
			res = self.relu(self.bn_skip(self.skip(downs[self.downsample-1])))
			x = torch.cat((x, res), 1)
			x = self.conv_skip(x)
			x = self.bn2(x)
			x = self.relu(x)
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			x = self.upsample(x)

		return x

class CNN(nn.Module):
	def __init__(self, block, layers, skips):
		#self.in_channels = 8
		super(CNN, self).__init__()
		self.layer0 = self._make_layer(block, 32, layers[0], 0, 0)
		self.layer1 = self._make_layer(block, layers[0], layers[1], 0, 0)
		self.layer2 = self._make_layer(block, layers[1], layers[2], 0, 0)
		self.layer3 = self._make_layer(block, layers[2], layers[3], 0, 0)
		self.layer4 = self._make_layer(block, layers[3], layers[4], 0, 0)
		self.layer_0 = self._make_layer(block, layers[0], 3, 1, skips[0])
		self.layer_1 = self._make_layer(block, layers[1], layers[0], 2, skips[1])
		self.layer_2 = self._make_layer(block, layers[2], layers[1], 3, skips[2])
		self.layer_3 = self._make_layer(block, layers[3], layers[2], 4, skips[3])
		self.layer_4 = self._make_layer(block, layers[4], layers[3], 5, skips[4])
	def _make_layer(self, block, in_planes, out_planes, down, skip):
		layers = []
		layers.append(block(in_planes, out_planes, stride=1, downsample = down, skips = skip))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))
		x = self.layer_0(self.layer_1(self.layer_2(self.layer_3(self.layer_4(x)))))

		return x


def train(_iter):
	#global train_iter
	#global logger
	global inputs
	global targets
	global downs
	print(_iter)
	cnn.train()
	train_loss = 0
	correct = 0
	total = 0
	sigma = 1./30.
	label = targets
	inputs, targets = inputs.cuda(), targets.cuda()
	optimizer.zero_grad()
	tmp = inputs.data + sigma * torch.randn(inputs.shape).cuda()
	outputs = cnn(Variable(tmp))
	downs = []
	loss = loss_function(outputs, targets)
	loss.backward()
	optimizer.step()


	#train_loss += loss.data[0]
	#_, predicted = torch.max(outputs.data, 1)
	#total += targets.size(0)
	#correct += predicted.eq(targets.data).cpu().sum()
	#info = {'Train_loss' : loss.data[0], 'Train_Error%' : 100 - 100 * correct/total}
	#for tag, value in info.items():
	#	logger.scalar_summary(tag, value, train_iter)
	#train_iter += 1
	print('Train : Loss: %.3f' % (loss.data[0]))
	
	#return 0
def task2():
	return CNN(BasicBlock, [128, 128, 128, 128, 128], [4, 4, 4, 4, 4])

train_iter = 0
LR = 1
inputs = torch.FloatTensor(32, 512, 512).normal_(0, 0.1)
targets = cv2.imread('noise_image.png')
#cv2.imwrite('noise.jpg', np.array(m))
#print(np.array(m))

targets = np.transpose(targets, (2, 0, 1))
inputs = np.expand_dims(inputs, axis=0)
targets = np.expand_dims(targets, axis=0)
inputs = torch.Tensor(inputs)
targets = torch.Tensor(targets)
inputs, targets = Variable(inputs), Variable(targets)

cnn = task2()
cnn.cuda()
cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
loss_function = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=LR)
#logger = Logger('./logs')
def main():
	global LR
	for _iter in range(1, 1800):
		train(_iter)
		#test(epoch)
	tmp = cnn(inputs)
	
	img = tmp[0].data.cpu().numpy()
	img = np.transpose(img, (1, 2, 0))
	cv2.imwrite('output.jpg', np.array(img))
	
	

if __name__ == "__main__":
	main()

