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
		self.w0 = []
		self.w1 = [random.uniform(-1., 1.) for x in range(self.node)]

		for i in range(self.input_size):
			self.w0.append([random.uniform(-1., 1.) for x in range(self.node)])

		self.w0 = np.array(self.w0)
		self.w1 = np.array(self.w1)
		self.change_input = np.array([[0.0] * self.node for x in range(self.input_size)])
		self.change_out = np.array([0.0 for x in range(self.node)])

	def run(self, input):
		self.ai = np.array([1,1,1])
		self.ai[0] = input[0]
		self.ai[1] = input[1]
		self.ah = sigmoid(self.w0.T.dot(self.ai))
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
			delta_input = self.ai[i] * delta_hidden
			self.w0[i] += 	self.lr0 * self.change_input[i] + self.lr1 * delta_input
			self.change_input[i] = delta_input

	def train(self, pattern):
		for i in range(self.step):
			x = 0
			for p in pattern:
				error = p[1] - self.run(p[0])
				self.back_propagation(error)
				x += np.abs(error)
			#print(x)
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
	test = [
		[0,0],
		[0,1],
		[1,0],
		[1,1],
		]

	NN.test(test)

if __name__ == "__main__":
	main()
