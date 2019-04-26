import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# exp = 'maml-2dnav-ppd5'
# exp = 'maml-cartpole'
# exp = 'maml-contmountcar'
# exp = 'maml-acrobot'
# exp = 'maml-halfcheevel'
# exp = 'maml-antvel'




rand_baseline = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'rand_results/data_metalearned'))

uni_baseline = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'uni_results/data_metalearned'))

unirand_baseline = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'unirand_results/data_metalearned'))



plt.figure('K Shot Baselines for Task Sampling')

plt.errorbar([i for i in range(rand_baseline.shape[1])], torch.mean(rand_baseline, 0).tolist(), torch.std(rand_baseline, 0).tolist(), color=np.array([1.,0.,0.]), capsize=5, capthick=2, label='rand')
plt.errorbar([i for i in range(uni_baseline.shape[1])], torch.mean(uni_baseline, 0).tolist(), torch.std(uni_baseline, 0).tolist(), color=np.array([0.,1.,0.]), capsize=5, capthick=2, label='uni')
plt.errorbar([i for i in range(unirand_baseline.shape[1])], torch.mean(unirand_baseline, 0).tolist(), torch.std(unirand_baseline, 0).tolist(), color=np.array([0.,0.,1.]), capsize=5, capthick=2, label='unirand')


plt.xlabel('Gradient Descent Iteration Number')
plt.ylabel('Return')
plt.title('K Shot Baselines for Task Sampling')

plt.legend(loc='upper left')
plt.show()