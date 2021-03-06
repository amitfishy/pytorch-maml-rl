import numpy as np
import math

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

class Continuous_MountainCarVTEnv(Continuous_MountainCarEnv):
	def __init__(self, task = {}):
		super(Continuous_MountainCarVTEnv, self).__init__()
		self.task = task
		# self.goal_position = task.get('goal_position', 0.45)
		self.power = task.get('power', 0.0015)

	def sample_tasks(self, num_tasks, sampling_type, points_per_dim=-1):
		if sampling_type == 'rand':
			# goal_positions = self.np_random.uniform(-1.1, 0.5, size=(num_tasks,))
			powers = self.np_random.uniform(0.0005, 0.0025, size=(num_tasks,))

			tasks = [{'power': power} for power in powers]
		elif sampling_type == 'uni':
			assert int(num_tasks) == int(points_per_dim), 'Number of tasks(mbs) should match the points per dimension if using `uni`'
			# points_dim_gp = np.linspace(-1.1, 0.5, num=points_per_dim)
			points_dim_pw = np.linspace(0.0005, 0.0025, num=points_per_dim)

			# goal_positions = np.zeros(points_per_dim**2)
			# powers = np.zeros(points_per_dim**2)

			# counter = 0
			# for i in range(points_per_dim):
			# 	for j in range(points_per_dim):
			# 		goal_positions[counter] = points_dim_gp[i]
			# 		powers[counter] = points_dim_pw[j]
			# 		counter = counter + 1

			tasks = [{'power': power} for power in points_dim_pw]
		elif sampling_type == 'unirand':
			assert int(num_tasks) == int(points_per_dim), 'Number of tasks(mbs) should match the points per dimension if using `unirand`'
			# points_dim_gp, pdgoal_step = np.linspace(-1.1, 0.5, endpoint=False, retstep=True, num=points_per_dim)
			points_dim_pw, pdpow_step = np.linspace(0.0005, 0.0025, endpoint=False, retstep=True, num=points_per_dim)

			# goal_positions = np.zeros(points_per_dim**2)
			# powers = np.zeros(points_per_dim**2)

			# counter = 0
			# for i in range(points_per_dim):
			# 	for j in range(points_per_dim):
			# 		goal_positions[counter] = points_dim_gp[i] + np.random.uniform(0, pdgoal_step)
			# 		powers[counter] = points_dim_pw[j] + np.random.uniform(0, pdpow_step)
			# 		counter = counter + 1
			points_dim_pw = points_dim_pw + np.random.uniform(0, pdpow_step, size=points_dim_pw.shape)
			
			tasks = [{'power': power} for power in points_dim_pw]
		else:
			assert False, 'Sampling type should be `uni` or `rand` or `unirand`. Given: `{}`'.format(sampling_type)

		return tasks

	def reset_task(self, task):
		self.task = task
		# self.goal_position = task['goal_position']
		self.power = task['power']
		return

	def reset(self):
		self.state = np.array([-0.5, 0]).astype(np.float32).flatten()
		return self.state

	def step(self, action):
		position = self.state[0]
		velocity = self.state[1]
		force = min(max(action[0], -1.0), 1.0)

		velocity += force*self.power -(self.power + 0.0010) * math.cos(3*position)
		if (velocity > self.max_speed): velocity = self.max_speed
		if (velocity < -self.max_speed): velocity = -self.max_speed
		position += velocity
		if (position > self.max_position): position = self.max_position
		if (position < self.min_position): position = self.min_position
		if (position==self.min_position and velocity<0): velocity = 0
		if (position==self.max_position and velocity>0): velocity = 0

		# if self.goal_position > -0.5:
		# 	done = bool(position >= self.goal_position)
		# else:
		# 	done = bool(position <= self.goal_position)

		done = bool(position >= self.goal_position)

		reward = 0
		if done:
			reward = 100.0
		reward-= math.pow(action[0],2)*0.1

		self.state = np.array([position, velocity]).astype(np.float32).flatten()
		return self.state, reward, done, {}

