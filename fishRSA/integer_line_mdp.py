#Adapted from https://agentmodels.org/chapters/3a-mdp.html
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

actions = [-1,0,1]
def transition(state, action):
	return state + action

def utility(state, action):
	return 1. if state==1 else -1.

def expected_utility(state, action, time_left):
	r = utility(state, action)  #Immediate reward
	new_time_left = time_left - 1
	if new_time_left == 0:
		return r
	else: #Get expected value of next state
		#Get transition probs
	    next_state_marginal = Marginal(transition)(state, action)
	    pd, pv = next_state_marginal._dist_and_values()
	    next_state_probs = pd.probs
	    next_states = list(pv.values())
	    #Get next_action for each next state. 
	    #Could use next_state_marginal.enumerate_support to ignore impossible successor states, but eh
	    #We should also take expectation over action TODO
	    next_actions_dists = [act(ns, new_time_left) for ns in next_states]
	    next_state_values = []
	    for i in range(len(next_states)):
	    	#The expected value is the expected expected expected utility, with expectation taken over next actions
	    	nad, nav = next_actions_dists[i]._dist_and_values()
	    	nad = nad.probs
	    	nav = list(nav.values())
	    	potential_expected_utilities = [expected_utility(next_states[i], a, new_time_left) for a in nav]
	    	next_state_values.append(torch.dot(nad, torch.tensor(potential_expected_utilities)))
	    # next_state_values = torch.tensor(data=[expected_utility(next_states[i],next_actions[i], new_time_left) for i in range(len(next_states))], dtype=torch.float32)
	    # print("next_state_probs: {}".format(next_state_probs))
	    # print("next_state_values: {}".format(next_state_values))
	    return r + torch.dot(next_state_probs, torch.tensor(next_state_values))

@Marginal
def act(state, time_left):
	action = actions[pyro.sample('a_id', dist.Categorical(torch.ones(len(actions))))]
	eu = expected_utility(state, action, time_left)
	factor('action_utility',10.0 * eu)
	return action

start_state = 0
total_time = 6
pd, pv = act(start_state, total_time)._dist_and_values()
print(pd.probs) 

"""
When goal is +-1 and total_time > 1, works correctly. When goal is 0 and total_time = 2, gives [.25, .5, .25]. For total_time != 2, uniform. For goal = 2, uniform.
"""

