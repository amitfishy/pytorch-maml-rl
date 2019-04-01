from comet_ml import Experiment

import maml_rl.envs
import gym
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from maml_rl.testing.k_shot_testing import k_shot_tester

from tensorboardX import SummaryWriter

import time


experiment = Experiment(api_key="wD61O90Y34KruZHAyHH7bqWRw", project_name="comp767-maml", workspace="amitfishy")

def total_rewards(episodes_rewards, gamma, aggregation=torch.mean):
    discount_matrix = torch.ones_like(episodes_rewards[0])
    for i in range(discount_matrix.shape[0]):
        discount_matrix[i, :] = discount_matrix[i, :] * gamma**i
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards * discount_matrix, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def train_meta_learning_model(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])

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

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(args.num_batches):
        print ('Currently processing Batch: {}'.format(batch+1))

        task_sampling_time = time.time()
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
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

        # Tensorboard
        # writer.add_scalar('total_rewards/before_update',
        #     total_rewards([ep.rewards for ep, _ in episodes]), batch)
        # writer.add_scalar('total_rewards/after_update',
        #     total_rewards([ep.rewards for _, ep in episodes]), batch)
        experiment.log_metric("Avg Disc Reward Before Update (MetaLearned)", total_rewards([ep.rewards for ep, _ in episodes], args.gamma), batch+1)
        experiment.log_metric("Avg Disc Reward After Update (MetaLearned)", total_rewards([ep.rewards for _, ep in episodes], args.gamma), batch+1)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(metalearner.policy.state_dict(), f)

    return


def train_pretrained_model(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])

    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder + '_pretrained'))
    save_folder = './saves/{0}'.format(args.output_folder + '_pretrained')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
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

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(args.num_batches):
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
        experiment.log_metric("Avg Disc Reward (Pretrained)", total_rewards([episodes.rewards], args.gamma), batch+1)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(metalearner.policy.state_dict(), f)

    return

def k_shot_experiments(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])


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

    save_folder_pretrained = './saves/{0}'.format(args.output_folder + '_pretrained')
    pretrained_model = os.path.join(save_folder_pretrained, 'policy-{0}.pt'.format(args.num_batches-1))
    policy_pretrained.load_state_dict(torch.load(pretrained_model))
    
    save_folder_metalearned = './saves/{0}'.format(args.output_folder + '_metalearned')
    metalearned_model = os.path.join(save_folder_metalearned, 'policy-{0}.pt'.format(args.num_batches-1))
    policy_metalearned.load_state_dict(torch.load(metalearned_model))

    pretrained_tester = k_shot_tester(args.K, policy_pretrained, args.K_shot_batch_size, 'Pretrained', args)
    avg_discounted_returns_pretrained = pretrained_tester.run_k_shot_exp()
    metalearned_tester = k_shot_tester(args.K, policy_metalearned, args.K_shot_batch_size, 'MetaLearned', args)
    avg_discounted_returns_metalearned = metalearned_tester.run_k_shot_exp()
    random_tester = k_shot_tester(args.K, policy_random, args.K_shot_batch_size, 'Random', args)
    avg_discounted_returns_random = random_tester.run_k_shot_exp()

    plt.figure('K Shot Testing Curves')
    plt.plot([i for i in range(args.K + 1)], avg_discounted_returns_pretrained, color=np.array([0.,0.,1.]), label='Pre-Trained')
    plt.plot([i for i in range(args.K + 1)], avg_discounted_returns_metalearned, color=np.array([0.,1.,0.]), label='Meta-Learned')
    plt.plot([i for i in range(args.K + 1)], avg_discounted_returns_random, color=np.array([0.,0.,0.]), label='Random')
    plt.ylabel('K Shot Iteration Number')
    plt.xlabel('Average Discounted Return')
    plt.title('K Shot Testing Curves')
    plt.legend(loc='upper left')
    plt.show()

    return


def main(args):
    if args.exp_type == 'MAML':
        #Get a starting state using MAML
        train_meta_learning_model(args)
        #Get a starting state using Pretraining
        train_pretrained_model(args)

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





    # General Arguments for K shot testing
    # Note that the exp type should be KSHOT for it
    parser.add_argument('--K', type=int, default=5,
        help='K shots for testing')
    parser.add_argument('--K-shot-batch-size', type=int, default=20,
        help='Number of episodes in each shot')




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
