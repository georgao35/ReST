from env.mujoco.humanoid_env import HumanoidEnv
from env.mujoco.ant_env import AntEnv
from env.mujoco.half_cheetah_env import HalfCheetahEnv

import gym
from gym.envs.registration import register
register(
    id='Humanoid-v4',
    entry_point='env.mujoco.humanoid_env:HumanoidEnv',
    kwargs={
        'render_hw': 320,
        'exclude_current_positions_from_observation': False
    }
)

register(
    id='Ant-v4',
    entry_point='env.mujoco.ant_env:AntEnv',
    kwargs={
        'render_hw': 320,
        'exclude_current_positions_from_observation': False
    }
)

register(
    id='HalfCheetah-v4',
    entry_point='env.mujoco.half_cheetah_env:HalfCheetahEnv',
    kwargs={
        'render_hw': 320,
        'exclude_current_positions_from_observation': False
    }
)
