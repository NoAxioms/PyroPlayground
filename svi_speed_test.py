import os, json, collections, math
import time
import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, Importance, EmpiricalMarginal, infer_discrete, config_enumerate
from pyro.optim import Adam
from search_inference import factor, HashingMarginal, memoize, Search, Marginal
from utilities import bayes_rule
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

num_states = 4
obs_noise = 0.1
obs_mat = torch.eye(num_states) * (1.0 - obs_noise) + torch.full((num_states,num_states), obs_noise/num_states)
init_belief_dist = dist.Categorical(torch.ones(num_states) / num_states)

def model(latent_prior):
	s = pyro.sample('latent', latent_prior)
	o = pyro.sample('observation', dist.Categorical(obs_mat[s]))
	return o

def guide(latent_prior, o):
	pass

def bayes_rule_test(n = 1000):
	init_belief = init_belief_dist.probs.numpy()
	obs = obs_mat.numpy()
	bayes_start = time.time()
	for i in range(n):
		b = bayes_rule(init_belief, obs)
	return time.time() - bayes_start

def svi_test(n = 1000):


bayes_duration = bayes_rule_test()
print("bayes_duration: ",bayes_duration)