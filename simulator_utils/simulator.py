from env.mujocoenvs import VecMuJoCoEnv
import torch
from simulator_utils.buffer import Buffer
from typing import Any, Dict, Tuple, List
from trainer_utils.agent import MlpActorCritic
from datetime import datetime
from tqdm import tqdm
from trainer_utils.collection import MlpActorCriticCollection

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class Simulator:
    def __init__(self, config: Dict[str, Any], expname: str) -> None:

        self.num_envs = config['num_envs']

        if expname == 'halfcheetah':
            self.env = VecMuJoCoEnv(self.num_envs, 'HalfCheetah-v3')
            self.exclude = 1
        elif expname == 'ant':
            self.env = VecMuJoCoEnv(self.num_envs, 'Ant-v3')
            self.exclude = 2
        elif expname == 'hopper':
            self.env = VecMuJoCoEnv(self.num_envs, 'Hopper-v3')
            self.exclude = 1
        elif expname == 'humanoid':
            self.env = VecMuJoCoEnv(self.num_envs, 'Humanoid-v3')
            self.exclude = 2
        elif expname == 'walker':
            self.env = VecMuJoCoEnv(self.num_envs, 'Walker2d-v3')
            self.exclude = 1
        elif expname == 'humanoid-v4':
            self.env = VecMuJoCoEnv(self.num_envs, 'Humanoid-v4')
            self.exclude = 2
        elif expname == 'ant-v4':
            self.env = VecMuJoCoEnv(self.num_envs, 'Ant-v4')
            self.exclude = 2
        elif expname == 'hc-v4':
            self.env = VecMuJoCoEnv(self.num_envs, 'HalfCheetah-v4')
            self.exclude = 1
        else:
            raise NotImplementedError

        self.obs_dim, self.act_dim = self.env.get_dims()
        self.traj_len = config['traj_len']
        self.gamma = config['gamma']
        self.lam = config['gaelam']

        self.buffer = Buffer(self.obs_dim, self.act_dim, self.num_envs, self.traj_len,
                             self.gamma, self.lam)

    def get_info(self) -> Tuple[Any, Any, Any]:
        return self.obs_dim, self.act_dim, self.exclude

    def run_sim(
            self, agent: MlpActorCritic, collector: MlpActorCriticCollection
    ) -> Tuple[Any, List[Any], List[Any]]:
        agent.eval()
        ep_ret_list = []
        ep_len_list = []
        ep_ret = torch.zeros(self.num_envs,
                             dtype=torch.float32,
                             device='cuda:0')
        ep_len = torch.zeros(self.num_envs,
                             dtype=torch.float32,
                             device='cuda:0')
        obs = self.env.reset()
        for t in tqdm(range(self.traj_len)):
            a, mu, std, logp, v = agent.step(obs)
            next_obs, _, done, info = self.env.step(a)

            with torch.no_grad():
                r = collector.get_intrinsic_reward(next_obs)

            ep_ret += r
            ep_len += 1
            self.buffer.store(obs, a, mu, std, r, v, logp, done)
            if done.any():
                env_ids = torch.nonzero(done,
                                        as_tuple=True)[0].to(dtype=torch.int32)
                for id in env_ids:
                    ep_ret_list.append(ep_ret[id].item())
                    ep_len_list.append(ep_len[id].item())
                ep_ret[env_ids.to(dtype=torch.long)] = 0
                ep_len[env_ids.to(dtype=torch.long)] = 0
                next_obs[env_ids.to(dtype=torch.long)] = self.env.reset(env_ids)[env_ids.to(dtype=torch.long)]
            obs = next_obs

        env_ids = torch.nonzero(~done, as_tuple=True)[0]
        for id in env_ids:
            ep_ret_list.append(ep_ret[id].item())
            ep_len_list.append(ep_len[id].item())

        _, _, _, _, v = agent.step(obs)
        self.buffer.finish_path(last_val=v)
        self.buffer.compute_adv()
        data = self.buffer.get()
        return data, ep_ret_list, ep_len_list

    def clear(self) -> None:
        self.buffer.clear()
