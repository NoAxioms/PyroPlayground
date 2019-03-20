import os
# import json
# import collections
# import math
#
# import numpy as np
# import torch
# import torchvision.datasets as dset
# import torch.nn as nn
# import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
# import pyro.poutine as poutine
# from pyro.infer import SVI, Trace_ELBO, Importance, EmpiricalMarginal, infer_discrete, config_enumerate
# from pyro.optim import Adam
#
# from search_inference import factor, HashingMarginal, memoize, Search, Marginal

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ
"""
Write compenents that work in simulated environments. Control the agent by constructing a Brooks esque control architecture.
Run SVI have components act based on inferred behavior about other components/agents as an observation

Use something like Pyro's sites to implement behavior interference?
Use DAN to simulate behaviourism?
"""

# from collections import namedtuple
# Observation = namedtuple('Observation', ["price", "amount_stored"])
def generate_observation():
	return {
		"price":pyro.sample('price', dist.Uniform(0.0,100.0))
	}

#Define trading behavior
class Behavior():
	def __init__(self):
		pass

	def run(self, observation):
		pass

class Buy_low_sell_high(Behavior):
	def __init__(self, pivot):
		self.pivot=pivot
	def run(self, observation):
		profit = observation.price - self.pivot
		return 1.0 / profit if profit != 0.0 else 0.0

#Define consrvation behavior
class Conserver(Behavior):
	def __init__(self, threshold):
		self.threshold = threshold
	def run(self, observation):
		return 0.0 if observation.amount_stored <= self.threshold else 1.0

class Merchant():
	def __init__(self):
		# self.nodes = [Conserver(0), Buy_low_sell_high(50)]
		# self.edges = [0, 1]
		self.conserver, self.buy_low_sell_high = Conserver(0), Buy_low_sell_high(50)
	def run(self):



#Write network structure
