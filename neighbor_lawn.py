import os
import json
import collections
import math

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoGuide
import torch.distributions.constraints as constraints
from search_inference import factor, HashingMarginal, memoize, Search, Marginal

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ


def lawn(rain_prob, my_sprinkler_prob, neighbor_sprinkler_prob):
	anomaly_rate=.001
	rain = pyro.sample("rain", dist.Bernoulli(rain_prob))
	#My lawn
	my_sprinkler = pyro.sample('my_sprinkler', dist.Bernoulli(my_sprinkler_prob))
	my_lawn = rain or my_sprinkler
	my_lawn_probs = [anomaly_rate, 1.0 - anomaly_rate] if my_lawn else [1.0 - anomaly_rate, anomaly_rate]
	my_lawn = pyro.sample('my_lawn', dist.Categorical(torch.tensor(my_lawn_probs)))
	#Neigbor lawn
	neighbor_sprinkler = pyro.sample('neighbor_sprinkler', dist.Bernoulli(neighbor_sprinkler_prob))
	neighbor_lawn = rain or neighbor_sprinkler
	neighbor_lawn_probs = [anomaly_rate, 1.0 - anomaly_rate] if neighbor_lawn else [1.0 - anomaly_rate, anomaly_rate]
	neighbor_lawn = pyro.sample('neighbor_lawn', dist.Categorical(torch.tensor(neighbor_lawn_probs)))
	return torch.tensor([my_lawn,neighbor_lawn])



def lawn_guide(rain_prob, my_sprinkler_prob, neighbor_sprinkler_prob):
	rain_prob = pyro.param('rain_prob', rain_prob, constraint=constraints.unit_interval)
	# print('rain_prob pre sample: ',rain_prob)
	my_sprinkler_prob = pyro.param('my_sprinkler_prob', my_sprinkler_prob, constraint=constraints.unit_interval)
	neighbor_sprinkler_prob = pyro.param('neighbor_sprinkler_prob', neighbor_sprinkler_prob, constraint=constraints.unit_interval)
	rain = pyro.sample('rain', dist.Bernoulli(rain_prob))
	my_sprinkler = pyro.sample('my_sprinkler', dist.Bernoulli(my_sprinkler_prob))
	neighbor_sprinkler = pyro.sample('neighbor_sprinkler', dist.Bernoulli(neighbor_sprinkler_prob))

def importance_empirical_test():
	conditioned_lawn = pyro.condition(lawn, data={"wet":torch.tensor([1.])})
	rain_post = pyro.infer.Importance(conditioned_lawn, num_samples=100)
	m = pyro.infer.EmpiricalMarginal(rain_post.run(), sites=["rain","sprinkler"])
	print(m.log_prob(torch.tensor([1.])))
	print(m.log_prob(torch.tensor([0.])))

def svi_test():
	rain_prob_prior = torch.tensor(.3)
	my_sprinkler_prob_prior = torch.tensor(.6)
	neighbor_sprinkler_prob_prior = torch.tensor(.2)
	conditioned_lawn = pyro.condition(lawn, data={"my_lawn":torch.tensor([1.]), "neighbor_lawn":torch.tensor([0.])})
	# guide = AutoGuide(lawn)
	# set up the optimizer
	adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
	optimizer = Adam(adam_params)

	# setup the inference algorithm
	svi = SVI(conditioned_lawn, lawn_guide, optimizer, loss=Trace_ELBO())

	n_steps = 1000
	# do gradient steps
	for step in range(n_steps):
		svi.step(rain_prob_prior, my_sprinkler_prob_prior, neighbor_sprinkler_prob_prior)
		if step % 100 == 0:
			print("step: ", step)
			for p in ['rain_prob', 'my_sprinkler_prob', 'neighbor_sprinkler_prob']:
				print(p, ": ", pyro.param(p).item())
	# for p in ['rain_prob', 'my_sprinkler_prob', 'neighbor_sprinkler_prob']:
	# 	print(p, ": ", pyro.param(p).item())

svi_test()