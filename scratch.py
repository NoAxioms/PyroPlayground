#Modified version of pyro's hyperbole.py, which is taken from https://gscontras.github.io/probLang/chapters/03-nonliteral.html
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


if __name__ == "__main__":
	flips = pyro.sample('flips', dist.Bernoulli(0.5 * torch.ones(10)))
	print(flips)