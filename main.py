from comet_ml import Experiment

import maml_rl.envs
import gym
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import pickle

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from maml_rl.testing.k_shot_testing import k_shot_tester

from tensorboardX import SummaryWriter

import time


#refer to comet ml for these 3 entries
api_key = 'put-key-here'
project_name = 'proj-name'
workspace = 'workspace'
experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace)

def total_rewards(episodes_rewards, gamma, aggregation=torch.mean):
    # discount_matrix = torch.ones(max([rewards.shape[0] for rewards in episodes_rewards]), episodes_rewards[0].shape[1]).to(episodes_rewards[0].device)
    # for i in range(discount_matrix.shape[0]):
    #     discount_matrix[i, :] = discount_matrix[i, :] * gamma**i
    # print('RSHAPE:',len(episodes_rewards))
    # print('DMSHAPE:',discount_matrix.shape)
    # for i,rewards in enumerate(episodes_rewards):
    #     print (i, rewards.shape)
    #     print (rewards)
    # rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards * discount_matrix[:rewards.shape[0], :rewards.shape[1]], dim=0))
    #     for rewards in episodes_rewards], dim=0))
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards , dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()



def train_meta_learning_model(args):
    # import matplotlib.pyplot as plt
    # import matplotlib.animation as animation
    # from matplotlib import style    

    # style.use('fivethirtyeight')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,1,1)    
    # xs = []
    # ys = []
    # def animate(i):
    #     ax1.clear()
    #     ax1.plot(xs, ys)
    rewards_before_ml = []
    rewards_after_ml = []

    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', 'MountainCarContinuousVT-v0'])

    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder + '_metalearned'))
    save_folder = './saves/{0}'.format(args.output_folder + '_metalearned')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    torch.manual_seed(args.random_seed)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    #load pretrained model
    cont_from_batch = 0
    if args.start_from_batch != -1:
        metalearned_model = os.path.join(save_folder, 'policy-{0}.pt'.format(args.start_from_batch-1))
        if os.path.exists(metalearned_model):
            policy.load_state_dict(torch.load(metalearned_model))
            cont_from_batch = args.start_from_batch

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(cont_from_batch, args.num_batches):
        print ('Currently processing Batch: {}'.format(batch+1))

        task_sampling_time = time.time()
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size, sampling_type=args.sampling_type, points_per_dim=args.points_per_dim)
        task_sampling_time = time.time() - task_sampling_time

        episode_generating_time = time.time()
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        episode_generating_time = time.time() - episode_generating_time

        learning_step_time = time.time()
        metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)
        learning_step_time = time.time() - learning_step_time

        print ('Tasking Sampling Time: {}'.format(task_sampling_time))
        print ('Episode Generating Time: {}'.format(episode_generating_time))
        print ('Learning Step Time: {}'.format(learning_step_time))
        reward_before_ml = total_rewards([ep.rewards for ep, _ in episodes], args.gamma)
        reward_after_ml = total_rewards([ep.rewards for _, ep in episodes], args.gamma)
        print ('Before Update: {} After Update: {}'.format(reward_before_ml, reward_after_ml))
        # experiment.log_metric("Avg Reward Before Update (MetaLearned)", reward_before_ml)
        experiment.log_metric("Avg Reward", reward_after_ml, batch+1)

        rewards_before_ml.append(reward_before_ml)
        rewards_after_ml.append(reward_after_ml)
        # xs.append(batch+1)
        # ys.append(total_rewards([ep.rewards for _, ep in episodes], args.gamma))
        # ani = animation.FuncAnimation(fig, animate, interval=1000)
        # plt.savefig('navg_baseline_monitor')
        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(metalearner.policy.state_dict(), f)

    # tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
    # episodes = metalearner.sample(tasks, first_order=args.first_order)
    # print("Avg Reward After Update (MetaLearned)", total_rewards([ep.rewards for _, ep in episodes], args.gamma))

    testing_sampler = BatchSampler(args.env_name, batch_size=args.testing_fbs,
        num_workers=args.num_workers)
    testing_metalearner = MetaLearner(testing_sampler, metalearner.policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)
    test_tasks = testing_sampler.sample_tasks(num_tasks=args.testing_mbs, sampling_type='rand', points_per_dim=-1)
    test_episodes = testing_metalearner.sample(test_tasks, first_order=args.first_order, no_update=True)
    test_reward = total_rewards([ep.rewards for ep in test_episodes], args.gamma)
    print('-------------------------------------------------')
    print ('Test Time reward is: ' + str(test_reward))
    print('-------------------------------------------------')

    pickle_reward_data_file = os.path.join(save_folder, 'reward_data.pkl')
    with open(pickle_reward_data_file, 'wb') as f:
        pickle.dump(rewards_before_ml, f)
        pickle.dump(rewards_after_ml, f)

    pickle_final_reward_file = os.path.join(save_folder, 'final_reward.pkl')
    with open(pickle_final_reward_file, 'wb') as f:
        pickle.dump(test_reward, f)

    return


def train_pretrained_model(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', 'MountainCarContinuousVT-v0'])

    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder + '_pretrained'))
    save_folder = './saves/{0}'.format(args.output_folder + '_pretrained')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    #batch_size=2*args.fast_batch_size to match the amount of data used in meta-learning
    sampler = BatchSampler(args.env_name, batch_size=2*args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    #load pretrained model
    cont_from_batch = 0
    if args.start_from_batch != -1:
        pretrained_model = os.path.join(save_folder, 'policy-{0}.pt'.format(args.start_from_batch-1))
        if os.path.exists(pretrained_model):
            policy.load_state_dict(torch.load(pretrained_model))
            cont_from_batch = args.start_from_batch

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(cont_from_batch, args.num_batches):
        print ('Currently processing Batch: {}'.format(batch+1))

        task_sampling_time = time.time()
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        task_sampling_time = time.time() - task_sampling_time

        episode_generating_time = time.time()
        episodes = metalearner.sample_for_pretraining(tasks, first_order=args.first_order)
        episode_generating_time = time.time() - episode_generating_time

        learning_step_time = time.time()
        params = metalearner.adapt(episodes, first_order=args.first_order)
        metalearner.policy.load_state_dict(params, strict=True)
        learning_step_time = time.time() - learning_step_time

        print ('Tasking Sampling Time: {}'.format(task_sampling_time))
        print ('Episode Generating Time: {}'.format(episode_generating_time))
        print ('Learning Step Time: {}'.format(learning_step_time))

        # Tensorboard
        # writer.add_scalar('total_rewards/before_update',
        #     total_rewards([ep.rewards for ep, _ in episodes]), batch)
        # writer.add_scalar('total_rewards/after_update',
        #     total_rewards([ep.rewards for _, ep in episodes]), batch)
        # experiment.log_metric("Avg Disc Reward (Pretrained)", total_rewards([episodes.rewards], args.gamma), batch+1)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(metalearner.policy.state_dict(), f)

    return

def k_shot_experiments(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', 'MountainCarContinuousVT-v0'])


    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy_pretrained = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
        policy_metalearned = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
        policy_random = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy_pretrained = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
        policy_metalearned = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
        policy_random = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)

    # save_folder_pretrained = './saves/{0}'.format(args.output_folder + '_pretrained')
    # pretrained_model = os.path.join(save_folder_pretrained, 'policy-{0}.pt'.format(args.num_batches-1))
    # policy_pretrained.load_state_dict(torch.load(pretrained_model))
    
    save_folder_metalearned = './saves/{0}'.format(args.output_folder + '_metalearned')
    metalearned_model = os.path.join(save_folder_metalearned, 'policy-{0}.pt'.format(args.num_batches-1))
    policy_metalearned.load_state_dict(torch.load(metalearned_model))

    # metalearned_tester = k_shot_tester(args.K_shot_batch_num, policy_metalearned, args.K_shot_batch_size, args.K_shot_num_tasks, 'MetaLearned', args)
    # avg_discounted_returns_metalearned = metalearned_tester.run_k_shot_exp()
    # print('Metalearned KSHOT result: ', avg_discounted_returns_metalearned)
    # print('Mean: ', torch.mean(avg_discounted_returns_metalearned, 0))
    results_folder = './saves/{0}'.format(args.output_folder + '_results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    kshot_fig_path1 = os.path.join(results_folder, 'kshot_testing')
    # kshot_fig_path2 = os.path.join(results_folder, 'ml_pre_diff')
    result_data_path = os.path.join(results_folder, 'data_')

    metalearned_tester = k_shot_tester(args.K_shot_batch_num, policy_metalearned, args.K_shot_batch_size, args.K_shot_num_tasks, 'MetaLearned', args)
    avg_discounted_returns_metalearned = metalearned_tester.run_k_shot_exp()
    # pretrained_tester = k_shot_tester(args.K_shot_batch_num, policy_pretrained, args.K_shot_batch_size, args.K_shot_num_tasks, 'Pretrained', args)
    # avg_discounted_returns_pretrained = pretrained_tester.run_k_shot_exp()

    # random_tester = k_shot_tester(args.K_shot_batch_num, policy_random, args.K_shot_batch_size, args.K_shot_num_tasks, 'Random', args)
    # avg_discounted_returns_random = random_tester.run_k_shot_exp()

    plt.figure('K Shot: Testing Curves')
    # plt.plot([i for i in range(args.K_shot_batch_num + 1)], avg_discounted_returns_pretrained, color=np.array([0.,0.,1.]), label='Pre-Trained')
    # plt.plot([i for i in range(args.K_shot_batch_num + 1)], avg_discounted_returns_metalearned, color=np.array([0.,1.,0.]), label='Meta-Learned')
    # plt.plot([i for i in range(args.K_shot_batch_num + 1)], avg_discounted_returns_random, color=np.array([0.,0.,0.]), label='Random')
    # plt.errorbar([i for i in range(args.K_shot_batch_num + 1)], torch.mean(avg_discounted_returns_pretrained, 0).tolist(), torch.std(avg_discounted_returns_pretrained, 0).tolist(), color=np.array([0.,0.,1.]), label='Pre-Trained', capsize=5, capthick=2)
    plt.errorbar([i for i in range(args.K_shot_batch_num + 1)], torch.mean(avg_discounted_returns_metalearned, 0).tolist(), torch.std(avg_discounted_returns_metalearned, 0).tolist(), color=np.array([0.,1.,0.]), label='Meta-Learned', capsize=5, capthick=2)
    # plt.errorbar([i for i in range(args.K_shot_batch_num + 1)], torch.mean(avg_discounted_returns_random, 0).tolist(), torch.std(avg_discounted_returns_random, 0).tolist(), color=np.array([0.,0.,0.]), label='Random', capsize=5, capthick=2)

    plt.xlabel('Gradient Descent Iteration Number')
    plt.ylabel('Average Discounted Return')
    plt.title('K Shot: Testing Curves')
    plt.legend(loc='upper left')
    plt.savefig(kshot_fig_path1)
    # plt.show()


    # plt.figure('K Shot: Difference between Metalearned and Pretrained')

    # plt.errorbar([i for i in range(args.K_shot_batch_num + 1)], torch.mean(avg_discounted_returns_metalearned-avg_discounted_returns_pretrained, 0).tolist(), torch.std(avg_discounted_returns_metalearned-avg_discounted_returns_pretrained, 0).tolist(), color=np.array([0.,0.,0.]), capsize=5, capthick=2)

    # plt.xlabel('Gradient Descent Iteration Number')
    # plt.ylabel('Average Discounted Return Difference')
    # plt.title('K Shot: Difference between Metalearned and Pretrained')
    # plt.savefig(kshot_fig_path2)
    # plt.show()

    #save torch tensor results to combine with other experiments
    # torch.save(avg_discounted_returns_pretrained, result_data_path + 'pretrained')
    torch.save(avg_discounted_returns_metalearned, result_data_path + 'metalearned')
    return


def main(args):
    if args.exp_type == 'MAML':
        #Get a starting state using MAML
        train_meta_learning_model(args)
        #Get a starting state using Pretraining
        # train_pretrained_model(args)

    if args.exp_type == 'KSHOT':
        k_shot_experiments(args)

    return

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    #Experiment Type
    parser.add_argument('--exp-type', type=str, default='MAML',
        help='Perform Meta Learning including baselines (MAML) or K-shot testing (KSHOT)')    

    # General Arguments for Meta Learning
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')
    parser.add_argument('--start-from-batch', type=int, default=-1,
        help='start training from this batch number - the model file should exist in save folder')


    #arguments for organized sampling
    parser.add_argument('--sampling-type', type=str, default='rand',
        help='Type of task sampling to be used: rand|uni')
    parser.add_argument('--points-per-dim', type=int, default=-1,
        help='Points along each dimension (if `uni`)')    


    #final meta-learned model testing params
    parser.add_argument('--testing-mbs', type=int, default=-1,
        help='Number of random tasks to sample for testing final Meta-Learned model')
    parser.add_argument('--testing-fbs', type=int, default=-1,
        help='Number of episodes in each random task to sample for testing final Meta-Learned model')      

    # General Arguments for K shot testing
    # Note that the exp type should be KSHOT for it
    parser.add_argument('--K-shot-batch-num', type=int, default=5,
        help='K shots for testing')
    parser.add_argument('--K-shot-batch-size', type=int, default=20,
        help='Number of episodes in each batch')
    parser.add_argument('--K-shot-num-tasks', type=int, default=50,
        help='Number of tasks to test over to get variance')

    #random seed
    parser.add_argument('--random-seed', type=int, default=47,
        help='Random Seed for torch init of policy network')

    args = parser.parse_args()

    hyper_params = {
    "env_name": args.env_name,
    "gamma": args.gamma,
    "tau": args.tau,
    "first_order": args.first_order,
    "hidden_size": args.hidden_size,
    "num_layers": args.num_layers,
    "fast_batch_size": args.fast_batch_size,
    "fast_lr": args.fast_lr,
    "num_batches": args.num_batches,
    "meta_batch_size": args.meta_batch_size,
    "max_kl": args.max_kl,
    "cg_iters": args.cg_iters,
    "cg_damping": args.cg_damping,
    "ls_max_steps": args.ls_max_steps,
    "ls_backtrack_ratio": args.ls_backtrack_ratio
    }
    experiment.log_parameters(hyper_params)
    
    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
