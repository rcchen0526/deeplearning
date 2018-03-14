import math
import random

class Neural:
	def __init__(self, pattern):
		self.node = 3
		self.w0 = [random.random() for x in range(self.node)]
		self.w1 = [random.random() for x in range(self.node)]

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
	print(NN.w0, NN.w1)
if __name__ == "__main__":
	main()
