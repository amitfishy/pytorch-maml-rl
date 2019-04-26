import numpy as np
import math

from gym.envs.classic_control.cartpole import CartPoleEnv

class CartPoleVTEnv(CartPoleEnv):
	def __init__(self, task = {}):
		super(CartPoleVTEnv, self).__init__()
		self.task = task
		self.force_mag = task.get('force_mag', 10)

	def sample_tasks(self, num_tasks, sampling_type, points_per_dim=-1):
		if sampling_type == 'rand':
			force_mags = self.np_random.uniform(7.5, 22.5, size=(num_tasks,))

			tasks = [{'force_mag': force_mag} for force_mag in force_mags]
		elif sampling_type == 'uni':
			assert int(num_tasks) == int(points_per_dim), 'Number of tasks(mbs) should match the points per dimension if using `uni`'
			force_mags = np.linspace(7.5, 22.5, num=points_per_dim)
			
			tasks = [{'force_mag': force_mag} for force_mag in force_mags]
		elif sampling_type == 'unirand':
			assert int(num_tasks) == int(points_per_dim), 'Number of tasks(mbs) should match the points per dimension if using `unirand`'
			force_mags, fm_step = np.linspace(7.5, 22.5, endpoint=False, retstep=True, num=points_per_dim)
			force_mags = force_mags + np.random.uniform(0, fm_step, size=force_mags.shape)

			tasks = [{'force_mag': force_mag} for force_mag in force_mags]
		else:
			assert False, 'Sampling Type should be `uni` or `rand` or `unirand`. Given: ' + sampling_type
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