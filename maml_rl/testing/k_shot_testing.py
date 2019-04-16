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
	def __init__(self, batch_num, policy, batch_size, num_tasks, k_shot_exp_name, args):
		self.batch_num = batch_num
		self.policy = policy
		#self.params = params
		self.batch_size = batch_size
		self.num_tasks = num_tasks
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
		tasks = self.sampler.sample_tasks(num_tasks=self.num_tasks)
		mean_discounted_returns_all_tasks = []
		starting_policy = self.policy
		print ()
		print ('K SHOT TESTING FOR: ' +  self.k_shot_exp_name)
		for num_task in range(self.num_tasks):
			print ('Currently processing Task Number: {}'.format(num_task+1))
			self.policy = starting_policy
			self.metalearner.policy = starting_policy
			self.sampler.reset_task(tasks[num_task])
			mean_discounted_returns_single_task = []

			for batch in range(self.batch_num):
				# print ('Currently processing Test Batch: {}'.format(batch+1))
				train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, device=self.device)
				self.params = self.metalearner.adapt(train_episodes, first_order=self.first_order)
				self.policy.load_state_dict(self.params, strict=True)
				self.metalearner.policy = self.policy
				#calculate discounted rewards (returns) from start
				mean_discounted_returns_single_task.append(self.get_mean_discounted_return(train_episodes.rewards))
			train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, device=self.device)
			mean_discounted_returns_single_task.append(self.get_mean_discounted_return(train_episodes.rewards))

			mean_discounted_returns_all_tasks.append(mean_discounted_returns_single_task)

		return torch.Tensor(mean_discounted_returns_all_tasks).to(self.device)
