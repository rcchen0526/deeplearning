import math
import random
import numpy as np

class Neural:
	def __init__(self, pattern):
		self.node = 3
		self.input_size = 3
		self.step = 500
		self.w0 = []
		self.w1 = []
		self.b0 = []
		self.b1 = []
		for i in range(self.input_size):
			self.w0.append([random.random() for x in range(self.node)])
			self.b0.append([random.random() for x in range(self.node)])

		self.w1.append([random.random() for x in range(self.node)])
		self.b1.append([random.random() for x in range(self.node)])
	def run(self, input):
		a0 = []
		for i in range(self.node):
			x = 0
			for j in range(self.input_size):
				x += w0[j][i] * input[j] + b0[j][i]
			a0.append(x)

		a1 = 0
		for i in range(self.node):
			a1 += w1[i] * a0[i] + b1[i]
# End of Class
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
	return y * (1 - y)

def main():
	pat = [
		[[0,0,0], [0]],
		[[0,0,1], [1]],
		[[0,1,0], [1]],
		[[1,0,0], [1]],
		[[0,1,1], [0]],
		[[1,0,1], [0]],
		[[1,1,0], [0]],
		[[1,1,1], [1]]
	]
	NN = Neural(pat)
	print(NN.w0)
	print(NN.w1)
if __name__ == "__main__":
	main()

	
	#a0		w00*a0+b00 + w10*a1+b10 + w20*a2+b20
#a1		w01*a0+b01 + w11*a1+b11 + w21*a2+b21
#a2		w02*a0+b02 + w12*a1+b11 + w21*a2+b22

#a 		w0*a0+b0 + w1*a1+b1 + w2*a2+b2
