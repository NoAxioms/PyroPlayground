#Taken from https://agentmodels.org/chapters/3-agents-as-programs.html
import os
import json

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

from search_inference import factor, HashingMarginal, memoize, Search, Marginal

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

states = ['hungry', 'pizza', 'steak frites']
actions = ['italian', 'french']

def action_prior():
	return actions[pyro.sample('a_id',dist.Categorical(probs=torch.ones([len(actions)])))]

def transition(state,action):
	if action == 'italian':
		return 'pizza'
	else:
		return 'steak frites'

@Marginal
def act(state, goal):
	action = action_prior()
	state_new = transition(state, action)
	factor('goal_achieved', 0. if state_new==goal else -999999.)
	return action

def run():
	state_init = 'hungry'
	goal = 'steak frites'
	action_marginal = act(state_init, goal)
	pd, pv = action_marginal._dist_and_values()
	print([(s, action_marginal.log_prob(s).exp().item())
		for s in action_marginal.enumerate_support()])

if __name__ == "__main__":
	run()