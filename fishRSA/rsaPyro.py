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

def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

def meaning(utterance, price):
    return utterance == price

@Marginal
def binary_listener(utterance, faces):
	"""
	return: item index sampled uniformly from faces matching utterance
	"""
	valid_faces_ids = [i for i in range(len(faces)) if utterance in faces[i]]
	probs = torch.zeros([len(faces)])
	probs[valid_faces_ids] = 1.0
	face_id = pyro.sample('item_id',dist.Categorical(probs=probs))
	return face_id.item()

@Marginal
def speaker(items,utterance_candidates, depth):
	"""
	return: index of utterance
	"""
	utterance_id = pyro.sample("utterance_id",dist.Categorical(probs=torch.ones([len(utterance_candidates)])))
	"""
	Build distribution of litener interpretation given utterance and softmax it
	"""

@Marginal
def pragmatic_listener(utterance, items, utterance_candidates, depth):


if __name__ == "__main__":
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(x.split(" ") for x in standard_faces["faces_classic"])
		faces_sym = tuple(x.split(" ") for x in standard_faces["faces_sym"])
	utterances = list(sorted(set((x for i in faces_classic for x in i))))
