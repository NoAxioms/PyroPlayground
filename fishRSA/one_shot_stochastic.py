#Taken from https://agentmodels.org/chapters/3-agents-as-programs.html
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

State = collections.namedtuple("State", ["dish", "table"])
init_state = State(dish="hungry",table='bad')
dishes = {"NA":"hungry", "italian":"pizza", "french":"steak frites"}
tables = ['bad', 'good', 'spectacular']
table_values = {"bad": -1., "good": 0., "spectacular": 1.}
# table_probs = {"NA":"bad", "italian":(.1,.3,.6), "french":(.6,.3,.1)}
actions = ['italian', 'french']

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def action_prior():
	return actions[pyro.sample('a_id',dist.Categorical(probs=torch.ones([len(actions)])))]

# def table_prior(action):
# 	return tables[pyro.sample('table_id',dist.Categorical(probs=torch.ones([len(tables)])))]
def get_table_dist(action):
	probs = {"NA":"bad", "italian":(.0,.1,.9), "french":(.9,.1,.0)}
	return dist.Categorical(probs=torch.tensor(probs[action]))
def transition(state,action):
	table_dist = get_table_dist(action)
	table = tables[pyro.sample('table_id', table_dist)]
	return State(dish=dishes[action], table=table)

def utility(state, desired_dish):
	if state.dish == 'hungry':
		dish_value = -10.
	elif state.dish == desired_dish:
		dish_value = 1.
	else:
		dish_value = -1.
	return dish_value + table_values[state.table]

def expected_utility(desired_dish, action, init_state=init_state):
	#This function is probably written poorly
	next_state_marginal = Marginal(transition)(init_state, action)
	pd, pv = next_state_marginal._dist_and_values()
	probs = pd.probs
	vals = torch.tensor([utility(s,desired_dish) for s in list(pv.values())])
	# print("probs: {}".format(probs))
	# print("vals: {}".format(vals))
	eu = torch.dot(probs, vals)
	return eu


@Marginal
def act(state, desired_dish, use_eu = True):
	"""
	Acts more deliberately using expected utility
	"""
	action = action_prior()
	state_new = transition(state, action)
	if use_eu:
		eu = expected_utility(desired_dish, action)
		factor('goal_achieved', 10 * eu)
	else:
		factor('goal_achieved', 10 * utility(state_new,desired_dish))
	return action

def run():
	state_init = State(dish='hungry',table='NA')
	desired_dish = 'steak frites'
	action_marginal = act(state_init, desired_dish)
	pd, pv = action_marginal._dist_and_values()
	print([(s, action_marginal.log_prob(s).exp().item())
		for s in action_marginal.enumerate_support()])

def expected_utility_test():
	action = "italian"
	desired_dish = "pizza"
	eu = expected_utility(desired_dish, action)
	print(eu)

if __name__ == "__main__":
	run()