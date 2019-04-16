import matplotlib.pyplot as plt
import torch
import numpy as np
import os

exp = 'maml-cont-mount-car-vt-nb20'

mbs80fbs10_pretrained = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'mbs80fbs10_results/data_pretrained'))
mbs80fbs10_metalearned = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'mbs80fbs10_results/data_metalearned'))

mbs40fbs20_pretrained = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'mbs40fbs20_results/data_pretrained'))
mbs40fbs20_metalearned = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'mbs40fbs20_results/data_metalearned'))

mbs20fbs40_pretrained = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'mbs20fbs40_results/data_pretrained'))
mbs20fbs40_metalearned = torch.load(os.path.join('/home/fishy2/anaconda3/envs/comp767_maml_project/code/pytorch-maml-rl/saves', exp, 'mbs20fbs40_results/data_metalearned'))


plt.figure('K Shot: Difference between Metalearned and Pretrained')

plt.errorbar([i for i in range(5 + 1)], torch.mean(mbs80fbs10_metalearned-mbs80fbs10_pretrained, 0).tolist(), torch.std(mbs80fbs10_metalearned-mbs80fbs10_pretrained, 0).tolist(), color=np.array([1.,0.,0.]), capsize=5, capthick=2, label='mbs80fbs10')
plt.errorbar([i for i in range(5 + 1)], torch.mean(mbs40fbs20_metalearned-mbs40fbs20_pretrained, 0).tolist(), torch.std(mbs40fbs20_metalearned-mbs40fbs20_pretrained, 0).tolist(), color=np.array([0.,1.,0.]), capsize=5, capthick=2, label='mbs40fbs20')
plt.errorbar([i for i in range(5 + 1)], torch.mean(mbs20fbs40_metalearned-mbs20fbs40_pretrained, 0).tolist(), torch.std(mbs20fbs40_metalearned-mbs20fbs40_pretrained, 0).tolist(), color=np.array([0.,0.,1.]), capsize=5, capthick=2, label='mbs20fbs40')

plt.xlabel('Gradient Descent Iteration Number')
plt.ylabel('Average Discounted Return Difference')
plt.title('K Shot: Difference between Metalearned and Pretrained')

plt.legend(loc='upper left')
plt.show()