from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'},
    max_episode_steps=200
)

register(
    'AntDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'},
    max_episode_steps=200
)

register(
    'AntPos-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'},
    max_episode_steps=200
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

# Variable Task Stuff
# ----------------------------------------

register(
    'CartPoleVT-v0',
    entry_point='maml_rl.envs.variable_tasks.cartpole:CartPoleVTEnv',
    max_episode_steps=100
)

register(
    'MountainCarContinuousVT-v0',
    entry_point='maml_rl.envs.variable_tasks.continuous_mountain_car:Continuous_MountainCarVTEnv',
    max_episode_steps=200
)

register(
    'AcrobotVT-v1',
    entry_point='maml_rl.envs.variable_tasks.acrobot:AcrobotVTEnv',
    max_episode_steps=200
)