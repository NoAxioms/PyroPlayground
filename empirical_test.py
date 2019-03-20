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

# emp_dist = dist.Empirical(torch.randn(2, 3, 10), torch.ones(2, 3))
p = .6
emp_dist = dist.Empirical(torch.tensor([0,1]), torch.tensor([math.log(p/(1-p)),math.log((1-p)/p)]))
probs = emp_dist._categorical.probs
print(probs)
# hist = {0:0, 1:0}
# n = 1000
# for _ in range(n):
# 	hist[emp_dist.sample().item()] += 1
# hist = {k:float(v)/n for k,v in hist.items()}
#
# print(hist)
# print({k:math.log(v) for k,v in hist.items()})