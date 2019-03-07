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

from search_inference import factor, HashingMarginal, memoize, Search, Marginal

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

tiger_locations = ['left', 'right']
agent_locations = tiger_locations + ['trapped', 'free']
State = collections.namedtuple("State",["tiger_location","agent_location"])
observations = ["left", "right", "silence"]
actions = ["go_left", "go_right", "listen"]
def transition(state, action):
	if action == "listen":
		return state
	if action == state:
		return "dead"
	else:
		return "free"

def utility(state, action):
	if state.tiger_location == state.agent_location:
		return -1000.
	if state.agent_location == "trapped":
		return -1.
	if state.agent_location == "free":
		return 10.

def sample_observation(state, action, roar_rate=.8, accuracy = .9):
	if action == "listen":
		hear_noise = pyro.sample('hear_noise', dist.Bernoulli(roar_rate))
		if hear_noise:
			roar_is_correct = pyro.sample('roar_is_correct',dist.Bernoulli(accuracy))
			return state.tiger_location if roar_is_correct else [s for s in tiger_locations if s != state][0]
	return "silence"

def initial_belief():
	tiger_location = tiger_locations[int(pyro.sample('tiger_location', dist.Bernoulli(0.5)))]
	return State(tiger_location=tiger_location, agent_location='trapped')

def update_belief(belief, action, observation):
	def new_belief():
		s_0 = belief()
		s_1 = transition(s_0, action)
		pyro.sample('obs', sample_observation, obs=observation)
		return s_1
	return new_belief

def run(state):
	#Condition on final state being "free", then infer actions
	t = 0
	action = actions[pyro.sample("action_id",dist.Categorical(torch.ones(len(actions))))]
	return transition(state,action)


def main():
	# start_state_id = pyro.sample("start_state_id",dist.Categorical(torch.ones([2])))
	b = initial_belief
	b1 = update_belief(b, 'listen', 'left')
	s = b1()
	print(s)

if __name__ == "__main__":
	main()