import numpy as np
import math

from gym.envs.classic_control.cartpole import CartPoleEnv

class CartPoleVTEnv(CartPoleEnv):
	def __init__(self, task = {}):
		super(CartPoleVTEnv, self).__init__()
		self.task = task
		self.force_mag = task.get('force_mag', 10)

	def sample_tasks(self, num_tasks):
		force_mags = self.np_random.uniform(7.5, 22.5, size=(num_tasks,))

		tasks = [{'force_mag': force_mag} for force_mag in force_mags]
		return tasks

	def reset_task(self, task):
		self.task = task
		self.force_mag = task['force_mag']
		return

	def reset(self):
		self.state = super(CartPoleVTEnv, self).reset().astype(np.float32).flatten()
		return self.state

	def step(self, action):
		state, reward, done, _ = super(CartPoleVTEnv, self).step(action)
		self.state = state.astype(np.float32).flatten()
		return self.state, reward, done, {}