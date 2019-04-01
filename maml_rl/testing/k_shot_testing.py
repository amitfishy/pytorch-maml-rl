import maml_rl.envs
import gym
import numpy as np
# import matplotlib.pyplot as plt

from maml_rl.metalearner import MetaLearner
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
import torch

class k_shot_tester(object):
	def __init__(self, k, policy, batch_size, k_shot_exp_name, args):
		self.k = k
		self.policy = policy
		#self.params = params
		self.batch_size = batch_size
		self.k_shot_exp_name = k_shot_exp_name
		self.first_order = args.first_order
		self.gamma = args.gamma

		self.sampler = BatchSampler(args.env_name, batch_size=self.batch_size, num_workers=args.num_workers)

		self.baseline = LinearFeatureBaseline(int(np.prod(self.sampler.envs.observation_space.shape)))

		self.metalearner = MetaLearner(self.sampler, self.policy, self.baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau, device=args.device)

		self.to(args.device)

	def to(self, device, **kwargs):
		self.policy.to(device, **kwargs)
		self.baseline.to(device, **kwargs)
		self.device = device
		return

	def get_mean_discounted_return(self, rewards):
		discount_matrix = torch.ones(rewards.shape)
		for i in range(discount_matrix.shape[0]):
			discount_matrix[i, :] = discount_matrix[i, :] * self.gamma**i

		cum_returns = torch.sum(discount_matrix*rewards.cpu(), 0)
		mean_discounted_return = torch.mean(cum_returns).item()
		#average_discounted_return_std = torch.std(cum_returns).item()

		return mean_discounted_return

	def run_k_shot_exp(self):
		task = self.sampler.sample_tasks(num_tasks=2)
		self.sampler.reset_task(task[0])

		mean_discounted_returns = []

		for batch in range(self.k):
			print ('Currently processing Test Batch: {}'.format(batch+1))

			train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, device=self.device)
			self.params = self.metalearner.adapt(train_episodes, first_order=self.first_order)
			self.policy.load_state_dict(self.params, strict=True)
			#calculate discounted rewards (returns) from start
			mean_discounted_returns.append(self.get_mean_discounted_return(train_episodes.rewards))

		train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, device=self.device)
		mean_discounted_returns.append(self.get_mean_discounted_return(train_episodes.rewards))

		# self.plot_stuff(avg_discounted_returns)
		return mean_discounted_returns
