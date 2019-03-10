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
from pyro.infer import SVI, Trace_ELBO, Importance, EmpiricalMarginal, infer_discrete, config_enumerate
from pyro.optim import Adam

from search_inference import factor, HashingMarginal, memoize, Search, Marginal

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

state_names = ['left', 'right']
states = [left_tiger, right_tiger] = torch.arange(2,dtype=torch.int)
# agent_locations = tiger_locations + ['trapped', 'free']
# State = collections.namedtuple("State",["tiger_location","agent_location"])
observations_names = ["left", "right", "silence"]
observations = [roar_left, roar_right, silence] = torch.arange(3,dtype=torch.int)
action_names = ["go_left", "go_right", "listen"]
actions =[go_left, go_right, listen] = torch.arange(3,dtype=torch.int)
initial_belief = dist.Bernoulli(0.5)
roar_rate, roar_accuracy = .8, 1.0
discount = 0.99
# def transition(state, action):
# 	return state

def get_transition_dist(state, action):
	probs = torch.zeros(states.shape[0])
	probs[state.item()] = 1.
	return dist.Categorical(probs=probs)

def check_terminal(state, action):
	return action != listen

def get_reward(state, action):
	if (state.item() == left_tiger.item() and action.item() == go_left.item()) or (state.item() == right_tiger.item() and action.item() == go_right.item()):
		return -1000.  #Tiger food
	elif action.item() == listen.item():
		return -1.  #Dank cell
	else:
		return 10.  #Freedom!

#Memoize
def get_observation_dist(state, action):
	if action == listen:
		if state == left_tiger:
			left_prob = roar_accuracy * roar_rate
			right_prob = (1.0 - roar_accuracy) * roar_rate
		else:
			right_prob = roar_accuracy * roar_rate
			left_prob = (1.0 - roar_accuracy) * roar_rate
		probs = torch.tensor([left_prob,right_prob,1.0-roar_rate])
	else:
		probs = torch.tensor([0.,0.,1.])
	return dist.Categorical(probs)

def sample_observation_from_belief(belief, action):
	state = pyro.sample('state', belief)
	obs_dist = get_observation_dist(state, action)
	return pyro.sample('observation', obs_dist)

def sample_state_bao(belief, action, observation):
	state0 = pyro.sample("state0", belief).int()
	state1 = pyro.sample('state1', get_transition_dist(state0,action)).int()
	o = pyro.sample('observation', get_observation_dist(state1,action), obs=observation)
	return state1
	
def update_belief(belief, action, observation):
	posterior = pyro.infer.Importance(sample_state_bao, num_samples=10)
	return pyro.infer.EmpiricalMarginal(posterior.run(initial_belief,listen,left_tiger),sites=['state1'])


def test_update_belief():
	b = update_belief(initial_belief, listen, roar_left)
	for i in range(10):
		s = b.sample()
		print(s)
		print(left_tiger)
	b = update_belief(initial_belief, listen, roar_right)
	for i in range(10):
		assert pyro.sample('s_r_{}'.format(i), b) == right_tiger


# def bss(belief, depth):
# 	values = torch.empty(actions.shape)	
# 	for a in actions:

def q_value(belief, action, depth):
	start_state = pyro.sample('start_state_q', belief).int()
	reward = get_reward(start_state,action)
	if depth == 0 or check_terminal(start_state, action):
		future_value = 0.0
	else:
		next_state = pyro.sample('next_state_q', get_transition_dist(start_state,action)).int()
		observation = pyro.sample('observation_q', get_observation_dist(next_state, action)).int()
		next_belief = update_belief(belief, action, observation)
		future_q_values = torch.empty(actions.shape)
		for a in actions:
			future_q_values[a] = q_value(next_belief, action, depth - 1)
		future_value, best_action = future_q_values.max(0)
	return reward + discount * future_value

def q_value_test():
	ql = q_value(initial_belief, listen, 2)
	print(ql)
	q = q_value(initial_belief, go_left, 2)
	print(q)

def main():
	# start_state_id = pyro.sample("start_state_id",dist.Categorical(torch.ones([2])))
	b = initial_belief
	print("b: {}".format(b))
	b1 = update_belief(b, 'listen', 'left')
	print("b1: {}".format(b1))
	s = b1()
	print(s)

if __name__ == "__main__":
	q_value_test()