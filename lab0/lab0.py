import random
import numpy as np

class Neural:
	def __init__(self):
		self.input_size = 3#contain a bias
		self.node = 3
		#self.output_size = 1
		self.step = 40000
		self.lr0, self.lr1 = 0.5, 1
		self.lr = 1
		self.w0 = []
		self.w1 = [random.uniform(-1, 1) for x in range(self.node)]

		for i in range(self.input_size):
			self.w0.append([random.uniform(-1, 1) for x in range(self.node)])

		self.change_input = [[0.0] * self.node for x in range(self.input_size)]
		self.change_out = [0.0 for x in range(self.node)]

	def run(self, input):
		self.ai, self.ah, self.ao = [], [], 0.0

		for i in range(self.input_size - 1):
			self.ai.append(input[i])
		self.ai.append(1)

		for i in range(self.node):
			sum = 0
			for j in range(self.input_size):
				sum += self.w0[j][i] * self.ai[j]
			self.ah.append(sigmoid(sum))

		sum = 0
		for j in range(self.node):
			sum += self.w1[j] * self.ah[j]
		self.ao = sigmoid(sum)
		
		return self.ao

	def back_propagation(self, error):
		out_delta = (error) * dsigmoid(self.ao)
		for i in range(self.node):
			delta_weight = self.ah[i] * out_delta
			self.w1[i] += self.lr0*self.change_out[i] + self.lr1*delta_weight
			self.change_out[i]=delta_weight

		delta_hidden = [0.0 for x in range(self.node)]
		for i in range(self.node):
			error = 0.0
			error += self.w1[i] * out_delta
			delta_hidden[i]= error * dsigmoid(self.ah[i])

		for i in range(self.input_size):

			for j in range(self.node):
				delta_input = self.ai[i] * delta_hidden[j]
				self.w0[i][j] += self.lr0*self.change_input[i][j] + self.lr1*delta_input
				self.change_input[i][j]=delta_input

	def train(self, pattern):
		for i in range(self.step):
			for p in pattern:
				error = p[1] - self.run(p[0])
				self.back_propagation(error)
	def test(self, pattern):
		for p in pattern:
			print(p, self.run(p))

# End of Class
