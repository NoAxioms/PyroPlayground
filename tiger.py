import os, json

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

observations = ["left", "right", "silence"]
states = ["left", "right", "dead", "free"]
actions = ["left", "right", "listen"]
def sample_observation(state, action, roar_rate=.8, accuracy = .9):
	if action == "listen":
		hear_noise = pyro.sample('hear_noise', dist.Bernoulli(roar_rate))
		if hear_noise:
			roar_is_correct = pyro.sample('roar_is_correct',dist.Bernoulli(accuracy))
			return state if roar_is_correct else [s for s in ["left","right"] if s != state][0]
	return "silence"

def transition(state, action):
	if action == "listen":
		return state
	if action == state:
		return "dead"
	else:
		return "free"

def run(state):
	#Condition on final state being "free", then infer actions
	t = 0
	action = actions[pyro.sample("action_id",dist.Categorical(torch.ones(len(actions))))]
	return transition(state,action)


def main():
	start_state_id = pyro.sample("start_state_id",dist.Categorical(torch.ones([2])))
	"""
	Need to sample action
	"""