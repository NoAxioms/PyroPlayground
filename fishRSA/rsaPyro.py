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

from search_inference import factor, HashingMarginal, memoize, Search

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ


def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))


def meaning(utterance, face):
    return utterance in face


def face_prior(faces):
	f_id = pyro.sample('f_id', dist.Categorical(probs=torch.ones([len(faces)])))
	return faces[f_id]


def utterance_prior(utterance_candidates):
	probs = torch.ones([len(utterance_candidates)])
	u_id = pyro.sample('u_id', dist.Categorical(probs=probs))
	return utterance_candidates[u_id]


@Marginal
def literal_listener(utterance, faces):
	"""
	return: item index sampled uniformly from faces matching utterance
	"""
	face = face_prior(faces)
	factor("literal_meaning", 0. if meaning(utterance, face) else -999999.)
	return face


@Marginal
def speaker(face, faces, utterance_candidates, depth=None):
	"""
	return: index of utterance
	"""
	alpha = 1.
	utterance = utterance_prior(utterance_candidates)
	literal_marginal = literal_listener(utterance, faces)
	with poutine.scale(scale=torch.tensor(alpha)):
		pyro.sample('listener', literal_marginal, obs=face)
	return utterance


@Marginal
def pragmatic_listener(utterance, utterance_candidates, faces, depth=None):
	face = face_prior(faces)
	speaker_marginal = speaker(face, faces, utterance_candidates)
	pyro.sample("speaker", speaker_marginal, obs=utterance)
	return face


def main():
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(tuple(x.split(" "))
		                      for x in standard_faces["faces_classic"])
		faces_sym = tuple(x.split(" ") for x in standard_faces["faces_sym"])
	utterance_candidates = tuple(
	    list(sorted(set((x for i in faces_classic for x in i)))))
	utterance = "moustache"
	for x in [utterance, utterance_candidates, faces_classic]:
		print(type(x[0]))
	pragmatic_marginal = pragmatic_listener(
	    utterance, utterance_candidates, faces_classic)
	print(pragmatic_marginal)
	pd, pv = pragmatic_marginal._dist_and_values()
	print([(s, pragmatic_marginal.log_prob(s).exp().item())
		for s in pragmatic_marginal.enumerate_support()])

if __name__ == "__main__":
	main()
