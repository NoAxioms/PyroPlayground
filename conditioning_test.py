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

def foo():
	a = pyro.sample('a', dist.Categorical(torch.ones(2)))
	#DOES NOT WORK. Use distribution transform
	b = pyro.param("b", torch.tensor([a * 10]))
	c = b + 1
	return c

foo_c = pyro.condition(foo, data={"b":10})
print(foo_c())