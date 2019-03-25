#Modified version of pyro's hyperbole.py, which is taken from https://gscontras.github.io/probLang/chapters/03-nonliteral.html
import os
import json, time

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam

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
		self.softmax = nn.functional.softmax

	def forward(self, z):
		# assert z.dtype == torch.double
		hidden = self.sigmoid(self.fc1(z))
		# assert hidden.dtype == torch.double
		updated_belief = self.softmax(self.fc21(hidden), dim=0)
		return updated_belief

def train(epoch, model, training_data, optimizer):
	model.train()
	train_loss = 0
	#Replace with generated data
	for batch_idx, (literal_listener, b_0, b_1_correct) in enumerate(training_data):
		nn_input  = torch.cat((literal_listener,b_0)).to(device)
		optimizer.zero_grad()
		b_1_model = model(nn_input)
		loss = kl(b_1_model, b_1_correct.to(device))
		if loss <= 0:
			print("Negative kl:")
			print(b_1_model.data)
			print(b_1_correct.data)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if False and batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx, len(training_data),
				100. * batch_idx / len(training_data),
				loss.item()))

	# print('====> Epoch: {} Average loss: {:.4f}'.format(
	# 	  epoch, train_loss / len(training_data)))

def test_model(model, test_data):
	model.eval()
	test_loss = 0
	for batch_idx, (literal_listener, b_0, b_1_correct) in enumerate(test_data):
		nn_input  = torch.cat((literal_listener,b_0)).to(device)
		b_1_model = model(nn_input)
		loss = kl(b_1_model, b_1_correct.to(device))
		print("literal_listener:\n{}".format(literal_listener.data.cpu().numpy()))
		print("Correct / found belief:\n{}\n{}".format(b_1_correct.data.cpu().numpy(), b_1_model.data.cpu().numpy()))

def kl(P,Q):
	return (P * (P / Q).log()).sum()

def gather_training_data(items, depth):
	rsa = RSA(items=items, default_depth = depth, theta=5)
	rsa.run()
	data = []
	for u in range(rsa.num_utterances):
		example = [rsa.listener_probs_lit[u], rsa.listener_prior, rsa.listener_probs[u]]
		example = [torch.tensor(data=i, dtype=torch.float) for i in example]
		data.append(tuple(example))
		print(example)
	return data




def main():
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(tuple(x.split(" "))
							  for x in standard_faces["faces_classic"])
		faces_sym = tuple(x.split(" ") for x in standard_faces["faces_sym"])
	model = belief_update(len(faces_classic)).to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-1)
	training_data = gather_training_data(faces_classic, 5)
	for epoch in range(1, 100 + 1):
		train(epoch, model, training_data, optimizer)
	test_model(model, training_data)
if __name__ == "__main__":
	main()
		# test(epoch)
		# with torch.no_grad():
		#     sample = torch.randn(64, 20).to(device)
		#     sample = model.decode(sample).cpu()
		#     save_image(sample.view(64, 1, 28, 28),
		#                'results/sample_' + str(epoch) + '.png')
