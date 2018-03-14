import math
import random
import numpy as np

class Neural:
	def __init__(self):
		self.node = 3
		self.input_size = 3
		self.step = 500
		self.w0 = []
		self.w1 = [random.uniform(-1, 1) for x in range(self.node)]
		self.b0 = []
		self.b1 = [random.uniform(-1, 1) for x in range(self.node)]
		for i in range(self.input_size):
			self.w0.append([random.uniform(-1, 1) for x in range(self.node)])
			self.b0.append([random.uniform(-1, 1) for x in range(self.node)])

	def run(self, input):
		a0 = []
		for i in range(self.node):
			sum = 0
			for j in range(self.input_size):
				sum += self.w0[j][i] * input[j] + self.b0[j][i]
			a0.append(sigmoid(sum))

		sum = 0
		for i in range(self.node):
			sum += self.w1[i] * a0[i] + self.b1[i]
		return sigmoid(sum)

	def train(self, pattern):
		for i in range(self.step):
			for p in pattern:
				out = self.run(p[0])
				label = p[1]

# End of Class
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
	return y * (1 - y)

def main():
	pat = (
		[[0,0,0], [0]],
		[[0,0,1], [1]],
		[[0,1,0], [1]],
		[[1,0,0], [1]],
		[[0,1,1], [0]],
		[[1,0,1], [0]],
		[[1,1,0], [0]],
		[[1,1,1], [1]]
		)
	
	NN = Neural()
	print(NN.w0)
	print(NN.w1)
	print(NN.b0)
	print(NN.b1)
	NN.train(pat)

if __name__ == "__main__":
	main()

#a0		w00*a0+b00 + w10*a1+b10 + w20*a2+b20
#a1		w01*a0+b01 + w11*a1+b11 + w21*a2+b21
#a2		w02*a0+b02 + w12*a1+b11 + w21*a2+b22

#a 		w0*a0+b0 + w1*a1+b1 + w2*a2+b2
