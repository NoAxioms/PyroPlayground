#Adapted from https://agentmodels.org/chapters/3a-mdp.html
import os, time
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
goal_state = 3
discount = 0.99
def transition(state, action):
	return (state + action) % 4

def utility(state, action):
	return 1. if state==goal_state else -10.
	
@memoize
def expected_utility(state, action, time_left):
	r = utility(state, action)  #Immediate reward
	new_time_left = time_left - 1
	if new_time_left == 0:
		# print("{} {} {}".format(action, state, r))
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
	    	if next_states[i] == goal_state:
	    		next_state_values.append(torch.tensor(0.0))
    		else:
		    	#The expected value is the expected expected expected utility, with expectation taken over next actions
		    	nad, nav = next_actions_dists[i]._dist_and_values()
		    	nad = nad.probs
		    	nav = list(nav.values())
		    	potential_expected_utilities = [expected_utility(next_states[i], a, new_time_left) for a in nav]
		    	next_state_values.append(torch.dot(nad, torch.tensor(potential_expected_utilities)))
	    # next_state_values = torch.tensor(data=[expected_utility(next_states[i],next_actions[i], new_time_left) for i in range(len(next_states))], dtype=torch.float32)
	    # print("next_state_probs: {}".format(next_state_probs))
	    # print("next_state_values: {}".format(next_state_values))
	    return r + discount * torch.dot(next_state_probs, torch.tensor(next_state_values))

@Marginal
def act(state, time_left):
	action = actions[pyro.sample('a_id', dist.Categorical(torch.ones(len(actions))))]
	eu = expected_utility(state, action, time_left)
	factor('action_utility',1000.0 * eu)
	return action

start_state = 0  #If start == goal, the actions don't matter and agent may act weird
total_time = 7  #Can reach goal total_time - 1 distance away
start_time = time.time()
pd, pv = act(start_state, total_time)._dist_and_values()
total_time = time.time() - start_time
print("total_time: {}".format(total_time))
print(pd.probs) 

"""
When total_time = 1, only immediate reward is counted, so the value of the next state is ignored.
When total_time is 2, the value of the next state is taken into account
When goal is +-1 and total_time > 1, works correctly. When goal is 0 and total_time = 2, gives [.25, .5, .25]. For total_time != 2, uniform. For goal = 2, uniform.
"""

