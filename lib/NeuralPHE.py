import numpy as np
import math
import torch
import torch.nn as nn

from .utils import Model, Net
from random import sample

class NeuralPHEStruct:
	def __init__(self,
				 featureDimension,
				 lambda_,
				 a,
				 batch_size,
				 hidden_size,
				 n_layers,
				 training_window,
				 learning_rate,
				 epochs,
				 p=0.0,
				 use_cuda=False):
		self.d = featureDimension
		self.lambda_ = lambda_
		self.time = 0

		self.armFeatureVecs = {}
		self.armTrials = {}
		self.armCumReward = {}
		self.grad_approx = {}
		self.history = []
		self.B_ = []
		self.a = a

		self.L = batch_size

		# hidden size of the NN layers
		self.hidden_size = hidden_size
		# number of layers
		self.n_layers = n_layers

		# number of rewards in the training buffer
		self.training_window = training_window

		# NN parameters
		self.learning_rate = learning_rate
		self.epochs = epochs

		self.use_cuda = use_cuda
		if self.use_cuda:
			raise Exception(
				'Not yet CUDA compatible : TODO for later (not necessary to obtain good results')
		self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

		# dropout rate
		self.p = p

		# neural network
		self.model = Model(input_size=featureDimension,
						   hidden_size=self.hidden_size,
						   n_layers=self.n_layers,
						   p=self.p
						   ).to(self.device)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

	def updateParameters(self, article_picked, click):
		self.time += 1
		if article_picked.id not in self.armTrials:
			self.armTrials[article_picked.id] = 0
			self.armFeatureVecs[article_picked.id] = article_picked.featureVector
			self.armCumReward[article_picked.id] = 0
		self.armTrials[article_picked.id] += 1
		self.armCumReward[article_picked.id] += click

		self.history.append((article_picked.featureVector, click))

		# sample L observations from user's history
		tmp = np.array(self.history, dtype=object)
		idx = np.random.choice(len(tmp), self.L, replace=True)
		self.B_ = tmp[idx]

		for l in range(self.L):
			for j in range(self.a):
				pseudo_reward = np.random.binomial(1, 0.5)
				np.append(self.B_, [(article_picked.featureVector, pseudo_reward)])

		self.train()


	def predict(self, article_featureVectors):
		"""Predict reward.
		        """
		# eval mode
		self.model.eval()
		rewards = self.model.forward(
			torch.FloatTensor(article_featureVectors).to(self.device)
		).detach().squeeze()
		# print("rewards:", rewards)
		return rewards


	def getTheta(self):
		for armID, armCumReward in self.armCumReward.items():
			x = torch.FloatTensor(
				self.armFeatureVecs[armID].reshape(1, -1)
			).to(self.device)

			self.model.zero_grad()
			y = self.model(x)
			y.backward()

			self.grad_approx[armID] = torch.cat(
				[w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if
				 w.requires_grad]
			).to(self.device)
		return 0 # np.dot(np.linalg.inv(self.B+self.lambda_*np.identity(self.d)), self.f_noiseless)

	def train(self):
		featureVecs = [i[0] for i in self.B_]
		rewards = [j[1] for j in self.B_]
		# print("featureVecs:", featureVecs)
		# print("rewards:", rewards)
		x_train = torch.FloatTensor(featureVecs).to(self.device)
		y_train = torch.FloatTensor(rewards).squeeze().to(self.device)

		self.model.train()
		for _ in range(self.epochs):
			y_pred = self.model.forward(x_train).squeeze()
			loss = nn.MSELoss()(y_train, y_pred)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

class NeuralPHE:
	def __init__(self, dimension, lambda_, perturbationScale=1):
		self.dimension = dimension
		self.perturbationScale = perturbationScale
		self.lambda_ = lambda_

		self.users = {}

		self.CanEstimateUserPreference = False

	def decide(self, pool_articles, userID):
		if userID not in self.users:
			self.users[userID] = NeuralPHEStruct(self.dimension,
												 self.lambda_,
												 self.perturbationScale,
												 batch_size=64,
												 hidden_size=128,
												 n_layers=3,
												 training_window=100,
												 learning_rate=1e-2,
												 epochs=100,
												 p=0.1)

		maxPTA = float('-inf')
		articlePicked = None

		# featureVecs = [x.featureVector for x in pool_articles]
		# rewards = self.users[userID].getProb(featureVecs)

		for x in pool_articles:
			if x.id not in self.users[userID].armTrials:
				return x
			x_pta = self.users[userID].predict([x.featureVector])
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		# articlePicked = pool_articles[np.argmax(rewards)]
		# print(np.argmax(rewards))
		return articlePicked

	def updateParameters(self, article_picked, click, userID):
		self.users[userID].updateParameters(article_picked, click)

	def getTheta(self, userID):
		return self.users[userID].getTheta()