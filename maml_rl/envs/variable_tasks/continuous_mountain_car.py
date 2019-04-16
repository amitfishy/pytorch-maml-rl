import numpy as np
import math

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

class Continuous_MountainCarVTEnv(Continuous_MountainCarEnv):
	def __init__(self, task = {}):
		super(Continuous_MountainCarVTEnv, self).__init__()
		self.task = task
		self.goal_position = task.get('goal_position', 0.45)
		self.power = task.get('power', 0.0015)

	def sample_tasks(self, num_tasks):
		goal_positions = self.np_random.uniform(-1.1, 0.5, size=(num_tasks,))
		powers = self.np_random.uniform(0.0005, 0.0025, size=(num_tasks,))

		tasks = [{'goal_position': goal_position, 'power': power} for goal_position, power in zip(goal_positions, powers)]
		return tasks

	def reset_task(self, task):
		self.task = task
		self.goal_position = task['goal_position']
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

		if self.goal_position > -0.5:
			done = bool(position >= self.goal_position)
		else:
			done = bool(position <= self.goal_position)

		reward = 0
		if done:
			reward = 100.0
		reward-= math.pow(action[0],2)*0.1

		self.state = np.array([position, velocity]).astype(np.float32).flatten()
		return self.state, reward, done, {}

