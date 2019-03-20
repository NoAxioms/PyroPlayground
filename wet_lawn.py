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


def lawn(rain_prob, sprinkler_prob):
	rain = pyro.sample("rain", dist.Bernoulli(rain_prob))
	sprinkler = pyro.sample('sprinkler', dist.Bernoulli(sprinkler_prob))
	wet = rain and sprinkler
	# wet=  pyro.sample('wet', dist.Delta(wet))
	if wet:
		wet = pyro.sample('wet', dist.Categorical(torch.tensor([.01, .99])))
	else:
		wet = pyro.sample('wet', dist.Categorical(torch.tensor([.99, .01])))
	return wet

def lawn_guide(rain_prob, sprinkler_prob):
	rain_prob = pyro.param('rain_prob', rain_prob, constraint=constraints.unit_interval)
	# print('rain_prob pre sample: ',rain_prob)
	sprinkler_prob = pyro.param('sprinkler_prob', sprinkler_prob, constraint=constraints.unit_interval)
	try:
		rain = pyro.sample('rain', dist.Bernoulli(rain_prob))
	except RuntimeError as e:
		print("rain_prob: {}".format(rain_prob))
		raise e
	sprinkler = pyro.sample('sprinkler', dist.Bernoulli(sprinkler_prob))

def importance_empirical_test():
	conditioned_lawn = pyro.condition(lawn, data={"wet":torch.tensor([1.])})
	rain_post = pyro.infer.Importance(conditioned_lawn, num_samples=100)
	m = pyro.infer.EmpiricalMarginal(rain_post.run(), sites=["rain","sprinkler"])
	print(m.log_prob(torch.tensor([1.])))
	print(m.log_prob(torch.tensor([0.])))

def svi_test():
	rain_prob_prior = torch.tensor(.3)
	sprinkler_prob_prior = torch.tensor(.6)
	conditioned_lawn = pyro.condition(lawn, data={"wet":torch.tensor([1.])})
	# guide = AutoGuide(lawn)
	# set up the optimizer
	adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
	optimizer = Adam(adam_params)

	# setup the inference algorithm
	svi = SVI(conditioned_lawn, lawn_guide, optimizer, loss=Trace_ELBO())

	n_steps = 5000
	# do gradient steps
	for step in range(n_steps):
		svi.step(rain_prob_prior, sprinkler_prob_prior)
		if step % 1000 == 0:
			print("step: ", step)
			for p in ['rain_prob', 'sprinkler_prob']:
				print(p, ": ", pyro.param(p).item())
	for p in ['rain_prob', 'sprinkler_prob']:
		print(p, ": ", pyro.param(p).item())

svi_test()