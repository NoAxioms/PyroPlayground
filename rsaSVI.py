#Modified version of pyro's hyperbole.py, which is taken from https://gscontras.github.io/probLang/chapters/03-nonliteral.html
import os
import json, time

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

from rsaClass import RSA
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

#Use Cuda
device = torch.device('cuda')

"""
Let belief be input vector, literal_listener probs/speaker probs be input vector, output deep listener probs
"""
class belief_update(nn.Module):
    def __init__(self, num_items):
        super(belief_update, self).__init__()
        self.fc1 = nn.Linear(num_items * 2, num_items * 2)
        self.fc21 = nn.Linear(num_items * 2, num_items)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.sigmoid(self.fc1(z))
        updated_belief = self.sigmoid(self.fc21(hidden))
        return updated_belief

def RSA_SVI(RSA):
	def __init__(self, **kwargs):
		super(RSA_SVI, self).__init__(**kwargs)
	def run(self, depth=None):
		if depth is None:
			depth = self.default_depth
	def speaker(depth):
		pass
	def listener(depth):
		pass


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def main():
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(tuple(x.split(" "))
		                      for x in standard_faces["faces_classic"])
		faces_sym = tuple(x.split(" ") for x in standard_faces["faces_sym"])
	rsa = RSA_SVI(items=faces_classic)
	print(rsa.listener_probs_list)
	# bu = belief_update()
	# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# main()
if __name__ == "__main__":
    for epoch in range(1, 100 + 1):
        train(epoch)
        # test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
