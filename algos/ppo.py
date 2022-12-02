import torch
from torch.optim import Adam
from algos.base import BasePolicy
from typing import Any, Dict, Tuple
from tqdm import tqdm


class PPO(BasePolicy):
    def __init__(
            self,
            config: Dict[str, Any],
            obs_dim: int,
            act_dim: int,
            num_envs: int,
            exclude: int
    ) -> None:
        super().__init__(
            config=config,
            obs_dim=obs_dim,
            act_dim=act_dim,
            independent=False,
            num_envs=num_envs,
            exclude=exclude
        )
        self.train_pi_iters = config['train_pi_iters']
        self.clip_ratio = config['clip_ratio']
        self.pi_optimizer = Adam(self.agent.actor.parameters(), lr=config['pi_lr'])

    def _compute_loss_pi(
            self,
            data: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, act, adv, logp = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        _, _, _, pi = self.agent.get_actor_output(obs)
        ent_loss = pi.entropy().mean()
        log_p = pi.log_prob(act).sum(axis=-1)
        ratio = torch.exp(log_p - logp)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        loss_pi -= ent_loss * 0.01

        # Useful extra info
        approx_kl = (logp - log_p).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def update_pi(
            self,
            data: Dict[str, Any]
    ) -> Any:
        for _ in tqdm(range(self.train_pi_iters)):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(data)

            loss_pi.backward()
            self.pi_optimizer.step()
        return loss_pi
