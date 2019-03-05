#first some imports
import torch
torch.set_default_dtype(torch.float64)  # double precision for numerical stability

import collections
import argparse
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from search_inference import factor, HashingMarginal, memoize, Search

def Marginal(fn):
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

@Marginal
def literal_listener(utterance):
    state = state_prior()
    factor("literal_meaning", 0. if meaning(utterance, state) else -999999.)
    return state

@Marginal
def speaker(state):
    alpha = 1.
    with poutine.scale(scale=torch.tensor(alpha)):
        utterance = utterance_prior()
        pyro.sample("listener", literal_listener(utterance), obs=state)
    return utterance

@Marginal
def pragmatic_listener(utterance):
    state = state_prior()
    pyro.sample("speaker", speaker(state), obs=utterance)
    return state

total_number = 4

def state_prior():
    n = pyro.sample("state", dist.Categorical(probs=torch.ones(total_number+1) / total_number+1))
    return n

def utterance_prior():
    ix = pyro.sample("utt", dist.Categorical(probs=torch.ones(3) / 3))
    return ["none","some","all"][ix]

meanings = {
    "none": lambda N: N==0,
    "some": lambda N: N>0,
    "all": lambda N: N==total_number,
}

def meaning(utterance, state):
    return meanings[utterance](state)

#silly plotting helper:
def plot_dist(d):
    support = d.enumerate_support()
    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    names = support

    ax = plt.subplot(111)
    width=0.3
    bins = map(lambda x: x-width/2,range(1,len(data)+1))
    ax.bar(bins,data,width=width)
    ax.set_xticks(map(lambda x: x, range(1,len(data)+1)))
    ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")

interp_dist = pragmatic_listener("some")
plot_dist(interp_dist)