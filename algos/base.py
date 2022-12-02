import torch
import torch.nn as nn
from torch.optim import Adam
from trainer_utils.agent import MlpActorCritic
from typing import Any, Dict
from tqdm import tqdm


class BasePolicy:
    def __init__(
            self,
            config: Dict[str, Any],
            obs_dim: int,
            act_dim: int,
            independent: bool,
            num_envs: int,
            exclude: int
    ) -> None:
        self.agent = MlpActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hid_sizes=config['hid_sizes'],
            activation=nn.Tanh,
            output_activation=nn.Identity,
            log_std_init=config['log_std_init'],
            independent=independent,
            num_envs=num_envs,
            exclude=exclude
        )
        self.agent = self.agent.to(torch.device('cuda:0'))
        self.l2_reg = 0
        self.train_v_iters = config['train_v_iters']
        self.vf_optimizer = Adam(self.agent.critic.parameters(), lr=config['vf_lr'])
        self.rnd_optimizer = Adam(self.agent.vae_module.parameters(), lr=1e-2)
        self.target_kl = config['target_kl']

    def _compute_loss_v(
            self,
            data: Dict[str, Any]
    ) -> torch.Tensor:
        obs, ret = data['obs'], data['ret']

        vf_loss = ((self.agent.get_critic_output(obs) - ret) ** 2).mean()

        return vf_loss

    def _update_v(
            self,
            data: Dict[str, Any]
    ) -> Any:
        optimizer = self.vf_optimizer

        for _ in tqdm(range(self.train_v_iters)):
            optimizer.zero_grad()
            loss_v = self._compute_loss_v(
                data=data
            )
            loss_v.backward()
            optimizer.step()
        return loss_v

    def _compute_loss_rnd(
            self,
            data: Dict[str, Any]
    ) -> torch.Tensor:
        obs = data['obs']

        # smoothing = torch.randn_like(obs, device='cuda:0') * 0.05
        # smoothing = torch.clamp(smoothing, -0.15, 0.15)
        # obs += smoothing

        rnd_loss = self.agent.get_rnd_module_output(obs)

        return rnd_loss

    def _update_rnd(
            self,
            data: Dict[str, Any]
    ) -> Any:
        optimizer = self.rnd_optimizer

        for _ in tqdm(range(self.train_v_iters)):
            optimizer.zero_grad()
            loss_rnd = self._compute_loss_rnd(
                data=data
            )
            loss_rnd.backward()
            optimizer.step()
            # print(loss_rnd)
        return loss_rnd

    def update_pi(
            self,
            data: Dict[str, Any]
    ) -> torch.tensor:
        raise NotImplementedError

    def update_params(
            self,
            data: Dict[str, Any],
            t: int
    ) -> Dict[str, Any]:

        self.agent.train()

        loss_pi = self.update_pi(
            data=data
        )

        loss_v = self._update_v(
            data=data
        )

        if t >= 0:
            loss_rnd = self._update_rnd(
                data=data
            )
        else:
            loss_rnd = -1

        return {'loss_pi': loss_pi, 'loss_v': loss_v, 'loss_rnd': loss_rnd}
