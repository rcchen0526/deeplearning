import math
import random

class Neural:
	def __init__(self, pattern):
		self.node = 3
		self.w0 = [random.random() for x in range(self.node)]
		self.w1 = [random.random() for x in range(self.node)]
