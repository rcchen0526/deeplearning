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

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.LeakyReLU(inplace=True)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		return out

class CNN(nn.Module):
	def __init__(self, block, layers):
		#self.in_channels = 8
		super(CNN, self).__init__()
		self.conv_b = nn.Sequential(	# input shape (3, 32, 32)
				nn.Conv2d(
				in_channels=3,		# input height
				out_channels=layers[0],	# n_filters
				kernel_size=3,		# filter size
				stride=1,		# filter movement/step
				padding=1,		#
				bias=False),				# output shape (64, 32, 32)
			#nn.ReLU(),		# activation
			#nn.MaxPool2d(kernel_size=2),	# output shape (64, 32, 32)
		)
		self.conv_e = nn.Sequential(	# input shape (3, 32, 32)
				nn.Conv2d(
				in_channels=layers[0],		# input height
				out_channels=3,	# n_filters
				kernel_size=3,		# filter size
				stride=1,		# filter movement/step
				padding=1,		#
				bias=False),				# output shape (64, 32, 32)
			#nn.ReLU(),		# activation
			#nn.MaxPool2d(kernel_size=2),	# output shape (64, 32, 32)
		)
		#downsample = D.Downsampler(n_planes=self.in_channels, factor=0, kernel_type='lanczos', phase=0.5, preserve_size=True)
		self.bn_b = nn.BatchNorm2d(layers[0])
		self.bn_e = nn.BatchNorm2d(3)
		self.relu = nn.LeakyReLU(inplace=True)
		self.layer1 = self._make_layer(block, layers[0], layers[1])
		self.layer2 = self._make_layer(block, layers[1], layers[2])
		self.layer3 = self._make_layer(block, layers[2], layers[3])
		self.layer4 = self._make_layer(block, layers[3], layers[4])
		self.layer_1 = self._make_layer(block, layers[1], layers[0])
		self.layer_2 = self._make_layer(block, layers[2], layers[1])
		self.layer_3 = self._make_layer(block, layers[3], layers[2])
		self.layer_4 = self._make_layer(block, layers[4], layers[3])
	def _make_layer(self, block, in_planes, out_planes, stride=1):
		layers = []
		layers.append(block(in_planes, out_planes, stride))
		return nn.Sequential(*layers)

	def forward(self, x):
		#print(x.shape)
		x = self.conv_b(x)
		#x = downsample(x)
		x = self.bn_b(x)
		x = self.relu(x)
		x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
		x = self.layer_1(self.layer_2(self.layer_3(self.layer_4(x))))
		x = self.conv_e(x)
		x = self.bn_e(x)
		x = self.relu(x)
		#Upsample
		return x


def train(_iter):
	#global train_iter
	#global logger
	global inputs
	global targets
	print(_iter)
	cnn.train()
	train_loss = 0
	correct = 0
	total = 0

	label = targets
	inputs, targets = inputs.cuda(), targets.cuda()
	optimizer.zero_grad()
	outputs = cnn(inputs)
	
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
def task1():
	return CNN(BasicBlock, [8, 16, 32, 64, 128])

train_iter = 0
LR = 1
inputs = torch.FloatTensor(3, 512, 512).normal_(0, 0.1)
targets = cv2.imread('noise_image.png')
#cv2.imwrite('noise.jpg', np.array(m))
#print(np.array(m))

targets = np.transpose(targets, (2, 0, 1))
inputs = np.expand_dims(inputs, axis=0)
targets = np.expand_dims(targets, axis=0)
inputs = torch.Tensor(inputs)
targets = torch.Tensor(targets)
inputs, targets = Variable(inputs), Variable(targets)

cnn = task1()
cnn.cuda()
cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
loss_function = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=LR)
#logger = Logger('./logs')
def main():
	global LR
	for _iter in range(1, 1200):
		train(_iter)
		#test(epoch)
	tmp = cnn(inputs)
	
	img = tmp[0].data.cpu().numpy()
	img = np.transpose(img, (1, 2, 0))
	cv2.imwrite('output.jpg', np.array(img))
	
	

if __name__ == "__main__":
	main()

