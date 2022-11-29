from typing import Any, Dict, List
import torch
from algos.ppo import PPO
import copy
import os


class MlpActorCriticCollection:
    def __init__(self, env_name: str, algo: str, config: Dict[str, Any],
                 obs_dim: int, act_dim: int, num_envs: int,
                 num_skills: int, exclude: int) -> None:
        self.trainer_list = []
        self.activation_mask = torch.zeros((num_skills),
                                           dtype=torch.float32,
                                           device='cuda:0')
        self.num_skills = num_skills
        self.last_deactivate_id = None
        self.num_envs = num_envs
        trainer = PPO(config, obs_dim, act_dim, num_envs=num_envs, exclude=exclude)
        arr = torch.arange(0, num_skills - 1, dtype=torch.float32, device='cuda:0')
        self.weight_mask = torch.exp(-arr * 0.5)
        self.weight_mask_sum = torch.sum(self.weight_mask)
        for _ in range(num_skills):
            skill_trainer = copy.deepcopy(trainer)  # PPO(config, obs_dim, act_dim, num_envs=num_envs)
            self.trainer_list.append(skill_trainer)

    def activate(self, skill_id: int) -> None:
        self.activation_mask[skill_id] = 1.

    def deactivate(self, skill_id: int) -> None:
        self.activation_mask[skill_id] = 0.
        if self.last_deactivate_id is not None:
            self.activation_mask[self.last_deactivate_id] = 1.
        self.last_deactivate_id = skill_id

    def reset(self, skill_id: int) -> None:
        self.trainer_list[skill_id].agent.rnd_module.reset()

    def get_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros((self.num_envs, self.num_skills),
                             dtype=torch.float32,
                             device='cuda:0')

        with torch.no_grad():
            for skill in range(self.num_skills):
                reward[:, skill] = self.trainer_list[skill].agent.get_rnd_module_reward(obs)
            reward = reward[:, self.activation_mask.to(dtype=torch.bool)]
            reward = torch.exp(-10 * reward)
            reward /= reward.shape[1]
            reward = torch.sum(reward, dim=1)
            reward = -torch.log(reward)
        return reward

    def save_model(self, save_dir: str) -> None:
        for skill_id in range(len(self.trainer_list)):
            skill_dir = os.path.join(save_dir, 'skill' + str(skill_id))
            os.mkdir(skill_dir)
            torch.save(self.trainer_list[skill_id].agent,
                       os.path.join(skill_dir, 'agent.pkl'))
            self.trainer_list[skill_id].agent.normalizer.save_model(
                os.path.join(skill_dir, 'normalizer'))
