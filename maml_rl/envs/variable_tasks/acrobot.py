import numpy as np
import math

from gym.envs.classic_control.acrobot import AcrobotEnv

class AcrobotVTEnv(AcrobotEnv):
	def __init__(self, task = {}):
		super(AcrobotVTEnv, self).__init__()
		self.task = task
		self.LINK_LENGTH_1 = task.get('LINK_LENGTH_1', 1.0)
		self.LINK_LENGTH_2 = task.get('LINK_LENGTH_2', 1.0)
		self.LINK_COM_POS_1 = self.LINK_LENGTH_1 / 2.0
		self.LINK_COM_POS_2 = self.LINK_LENGTH_2 / 2.0
		self.LINK_MASS_1 = task.get('LINK_MASS_1', 1.0)
		self.LINK_MASS_2 = task.get('LINK_MASS_2', 1.0)

	def sample_tasks(self, num_tasks):
		LINK_LENGTH_1s = self.np_random.uniform(0.6, 1.4, size=(num_tasks,))
		LINK_LENGTH_2s = self.np_random.uniform(0.6, 1.4, size=(num_tasks,))
		LINK_COM_POS_1s = LINK_LENGTH_1s / 2.0
		LINK_COM_POS_2s = LINK_LENGTH_2s / 2.0
		LINK_MASS_1s = self.np_random.uniform(0.6, 1.4, size=(num_tasks,))
		LINK_MASS_2s = self.np_random.uniform(0.6, 1.4, size=(num_tasks,))

		tasks = [{'LINK_LENGTH_1': LINK_LENGTH_1, 'LINK_LENGTH_2': LINK_LENGTH_2,
				'LINK_COM_POS_1': LINK_COM_POS_1, 'LINK_COM_POS_2': LINK_COM_POS_2,
				'LINK_MASS_1': LINK_MASS_1, 'LINK_MASS_2': LINK_MASS_2}
				for LINK_LENGTH_1, LINK_LENGTH_2, LINK_COM_POS_1, LINK_COM_POS_2, LINK_MASS_1, LINK_MASS_2 in
				zip(LINK_LENGTH_1s, LINK_LENGTH_2s, LINK_COM_POS_1s, LINK_COM_POS_2s, LINK_MASS_1s, LINK_MASS_2s)]
		
		return tasks

	def reset_task(self, task):
		self.task = task
		self.LINK_LENGTH_1 = task['LINK_LENGTH_1']
		self.LINK_LENGTH_2 = task['LINK_LENGTH_2']
		self.LINK_COM_POS_1 = task['LINK_COM_POS_1']
		self.LINK_MASS_1 = task['LINK_MASS_1']
		self.LINK_MASS_2 = task['LINK_MASS_2']
		return

	def reset(self):
		self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,)).astype(np.float32).flatten()
		return self._get_ob().astype(np.float32).flatten()

	def step(self, action):
		_, reward, done, _ = super(AcrobotVTEnv, self).step(action)
		self.state = self.state.astype(np.float32).flatten()
		return self._get_ob().astype(np.float32).flatten(), reward, done, {}

