import matplotlib.pyplot as plt
import random
import numpy as np

class Neural:
	def __init__(self):
		self.input_size = 3#contain a bias
		self.node = 3
		#self.output_size = 1
		self.step = 2000
		self.lr0, self.lr1 = 0.5, 1
		self.lr = 1.5
		self.costs = []
		self.w = []
		self.w0 = []
		self.w1 = [random.uniform(-1., 1.) for x in range(self.node)]

		for i in range(self.input_size):
			self.w.append([random.uniform(-1., 1.) for x in range(self.node)])
			self.w0.append([random.uniform(-1., 1.) for x in range(self.node)])

		self.w = np.array(self.w)
		self.w0 = np.array(self.w0)
		self.w1 = np.array(self.w1)
		self.change_input = np.array([[0.0] * self.node for x in range(self.input_size)])
		self.change_hid = np.array([[0.0] * self.node for x in range(self.input_size)])
		self.change_out = np.array([0.0 for x in range(self.node)])

	def run(self, input):
		self.ai = np.array([1,1,1])
		self.ai[0] = input[0]
		self.ai[1] = input[1]
		self.a0 = sigmoid(self.w.T.dot(self.ai))
		self.ah = sigmoid(self.w0.T.dot(self.a0))
		self.ao = sigmoid(self.w1.T.dot(self.ah))
		return self.ao

	def back_propagation(self, error):
		out_delta = (error) * dsigmoid(self.ao)
		
		delta_weight = self.ah * out_delta 
		self.w1 += self.lr0 * self.change_out + self.lr1 * delta_weight
		self.change_out = delta_weight
			
		delta_hidden = np.array([0.0 for x in range(self.node)])
		delta_hidden = self.w1 * out_delta * dsigmoid(self.ah)
		
		for i in range(self.input_size):
			delta_hid = self.a0[i] * delta_hidden
			self.w0[i] += self.lr0 * self.change_hid[i] + self.lr1 * delta_hid
			self.change_hid[i] = delta_hid

		delta = np.array([0.0 for x in range(self.node)])
		hidden = np.array([0.0 for x in range(self.node)])
		for j in range(self.node):
			error = self.w0[j].dot(delta_hidden)
			hidden[j] = error * dsigmoid(self.a0[j])
		
		for i in range(self.node):
			delta_input = hidden * self.ai[i]
			self.w[i] += self.lr0 * self.change_input[i] + self.lr1  * delta_input
			self.change_input[i] = delta_input

	def train(self, pattern):
		for i in range(self.step):
			x = 0
			for p in pattern:
				error = p[1] - self.run(p[0])
				self.back_propagation(error)
				x += np.abs(error)
			#print(x)
			self.costs.append(x)

	def test(self, pattern):
		for p in pattern:
			print(p, self.run(p))

# End of Class
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
	return x * (1 - x)

def main():
	pat = (
		[[0,0], [0]],
		[[1,1], [0]],
		[[1,0], [1]],
		[[0,1], [1]],
		)
	
	NN = Neural()
	NN.train(pat)
	plt.plot(range(NN.step), NN.costs)
	plt.show()
	test = [
		[0,0],
		[0,1],
		[1,0],
		[1,1],
		]

	NN.test(test)

if __name__ == "__main__":
	main()
