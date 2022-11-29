import gym
from gym import wrappers
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_env(env_name):
    return {
        'Walker2d': 'Walker2d-v3',
        'HalfCheetah': 'HalfCheetah-v3',
        'Hopper': 'Hopper',
    }[env_name]


parser = argparse.ArgumentParser()
parser.add_argument('--timestamp', type=str, help='the timestep where model is saved')
parser.add_argument('--env', type=str, help='same as the --env of training processes')
parser.add_argument('--epoch', type=int, default=49, help='the epoch when model is saved')
parser.add_argument('--skid', type=int, default=9, help='the number of skill that is being updated when model is saved')
parser.add_argument('--traj_len', type=int, default=1000, help='length of evaluation trajectory in each episode')
parser.add_argument('--num_skills', type=int, default=10, help='number of skills discovered')
parser.add_argument('--episode_per_skill', type=int, default=5, help='evaluated episodes of each skill')
args = parser.parse_args()

TIMESTAMP = args.timestamp
epoch = args.epoch
skid = args.skid
tstp = 19

num_per_skill = args.episode_per_skill
env = gym.make(get_env(args.env_name), exclude_current_positions_from_observation=False)
obs_dim = env.observation_space.shape[0]
trajs = np.zeros((args.num_skills, num_per_skill, args.traj_len, obs_dim))

plt.figure(figsize=(3, 2))
y_lim_up = 25
y_lim_bottom = -10
plt.ylim(y_lim_bottom, y_lim_up)

for j in range(num_per_skill):
    for skill in range(args.num_skills):
        env = gym.make(get_env(args.env_name), exclude_current_positions_from_observation=False)
        obs_dim = env.observation_space.shape[0]
        agent = torch.load(
                os.path.join(
                    './rnd_models_single_run/' + TIMESTAMP + '/' + str(epoch) + '/' + str(skid) + '/' + str(tstp) + '/skill' + str(skill),
                    'agent.pkl'))

        agent.normalizer.load_model(
                os.path.join(
                    './rnd_models_single_run/' + TIMESTAMP + '/' + str(epoch) + '/' + str(skid) + '/' + str(tstp) + '/skill' + str(skill),
                    'normalizer'))
        
        env = wrappers.Monitor(env, f'./videos/{skill}_{j}', force=True)

        agent.eval()

        obs = env.reset()
        obs = torch.from_numpy(obs).to(device='cuda:0')
        ep_ret = 0
        traj = []
        xy_traj = []
        if True:
            for i in tqdm(range(args.traj_len)):
                a, _, _, _, _ = agent.step(obs.unsqueeze(0))
                obs, r, d, info = env.step(a.cpu().numpy())
                traj.append(obs)
                trajs[skill, j, i, :] = obs
                obs = torch.from_numpy(obs).to(device='cuda:0')
                xy_traj.append(np.array([info['x_position']]))
                if d:
                    break
                ep_ret += r
            traj = np.stack(traj, axis=0)
            xy_traj = np.stack(xy_traj, axis=0)
        plt.plot(xy_traj[:, 0], linewidth=2)

    plt.gca().set_prop_cycle(None)
if not os.path.exists(f'./renders/{TIMESTAMP}_{epoch}'):
    os.mkdir(f'./renders/{TIMESTAMP}_{epoch}')
plt.savefig(f'./renders/{TIMESTAMP}_{epoch}_path_{y_lim_bottom}_{y_lim_up}.pdf', bbox_inches='tight')
plt.close()
