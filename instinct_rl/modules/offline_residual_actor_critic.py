from collections import OrderedDict
from copy import copy
from typing import Dict

import importlib
import torch
import torch.nn as nn
from torch.distributions import Normal

import instinct_rl.modules as instinct_modules
from instinct_rl.utils.utils import get_subobs_size

from .actor_critic import get_activation
from .residual_moe_actor_critic import FlatTrunkHead


class OfflineResidualActorCritic(nn.Module):
    """Cross-attention encoded flat-base actor with a single residual MLP.

    The actor computes ``a = a_flat + delta_a``.  The flat base is exposed as
    ``.base`` so flat-prior initialization and freezing can reuse the same path
    as the residual-MoE policies.
    """

    is_recurrent = False

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions,
        encoder_configs: Dict[str, dict],
        critic_encoder_configs=None,
        actor_hidden_dims=(256, 128, 128),
        residual_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(256, 128, 64),
        activation="elu",
        init_noise_std=1.0,
        num_rewards=1,
        residual_delta_scale=1.0,
        residual_output_init_std=1.0e-3,
        residual_action_clip=None,
        action_mean_clip=None,
        sanitize_nonfinite=True,
        mu_activation=None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "OfflineResidualActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.__obs_segments = obs_format["policy"]
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        self.encoder_configs = copy(encoder_configs)
        self.critic_encoder_configs = copy(critic_encoder_configs)
        self.num_actions = num_actions
        self.activation = activation
        self.mu_activation = mu_activation
        self.actor_hidden_dims = list(actor_hidden_dims)
        self.residual_hidden_dims = list(residual_hidden_dims)
        self.critic_hidden_dims = list(critic_hidden_dims)
        self.residual_delta_scale = residual_delta_scale
        self.residual_action_clip = residual_action_clip
        self.action_mean_clip = action_mean_clip
        self.sanitize_nonfinite = sanitize_nonfinite

        self.encoders = self._build_encoder(self.__obs_segments, self.encoder_configs)
        self.critic_encoders = None
        if self.critic_encoder_configs == "shared":
            self.critic_encoders = self.encoders
        elif self.critic_encoder_configs is not None:
            self.critic_encoders = self._build_encoder(self.__critic_obs_segments, self.critic_encoder_configs)

        self.encoded_segments = self.encoders.output_segment
        encoded_dim = get_subobs_size(self.encoded_segments)
        critic_segments = self.critic_encoders.output_segment if self.critic_encoders is not None else self.__critic_obs_segments
        critic_input_dim = get_subobs_size(critic_segments)

        self.base = FlatTrunkHead(encoded_dim, num_actions, self.actor_hidden_dims, activation)
        self.residual = self._build_mlp(encoded_dim, num_actions, self.residual_hidden_dims, activation)
        if self.mu_activation:
            self.mu_activation_fn = get_activation(self.mu_activation)
        else:
            self.mu_activation_fn = None
        self._init_residual_output(residual_output_init_std)

        if num_rewards > 1:
            self.critics = nn.ModuleList(
                [self._build_mlp(critic_input_dim, 1, self.critic_hidden_dims, activation) for _ in range(num_rewards)]
            )
        else:
            self.critic = self._build_mlp(critic_input_dim, 1, self.critic_hidden_dims, activation)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        self.last_base_action = None
        self.last_residual_action = None
        self.last_residual_mixture = None

        print(f"Actor Encoder: {self.encoders}")
        print(f"Critic Encoder: {self.critic_encoders}")
        print(f"Flat Base: {self.base}")
        print(f"Residual MLP: {self.residual}")
        if num_rewards > 1:
            print(f"Multiple Critics MLP: {len(self.critics)} in total.")
        else:
            print(f"Critic MLP: {self.critic}")

    def _sanitize(self, tensor, clip=None):
        if self.sanitize_nonfinite:
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None and clip > 0.0:
            tensor = torch.clamp(tensor, -clip, clip)
        return tensor

    @staticmethod
    def _build_encoder(input_segments: OrderedDict, encoder_configs: Dict[str, dict]):
        encoder_configs = copy(encoder_configs)
        encoder_class_name = encoder_configs.pop("class_name", "ParallelLayer")
        EncoderClass = (
            getattr(importlib.import_module(encoder_class_name.split(":")[0]), encoder_class_name.split(":")[1])
            if ":" in encoder_class_name
            else getattr(instinct_modules, encoder_class_name)
        )
        return EncoderClass(input_segments=input_segments, block_configs=encoder_configs)

    @staticmethod
    def _build_mlp(input_dim, output_dim, hidden_dims, activation):
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(get_activation(activation))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_residual_output(self, init_std):
        last_linear = [module for module in self.residual.modules() if isinstance(module, nn.Linear)][-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(last_linear.bias)

    def reset(self, dones=None):
        pass

    def _actor_forward_encoded(self, encoded_obs):
        encoded_obs = self._sanitize(encoded_obs)
        base_action = self.base(encoded_obs)
        residual_action = self.residual(encoded_obs) * self.residual_delta_scale
        residual_action = self._sanitize(residual_action, self.residual_action_clip)
        action = base_action + residual_action
        if self.mu_activation_fn is not None:
            action = self.mu_activation_fn(action)
        action = self._sanitize(action, self.action_mean_clip)
        self.last_base_action = base_action
        self.last_residual_action = residual_action
        self.last_residual_mixture = residual_action
        return action

    def forward(self, observations):
        return self.act_inference(observations)

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        std = self._sanitize(self.std, None).clamp_min(1.0e-6)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        observations = self._sanitize(observations)
        encoded_obs = self.encoders(observations)
        return self._actor_forward_encoded(encoded_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self._sanitize(critic_observations)
        obs = self.critic_encoders(critic_observations) if self.critic_encoders is not None else critic_observations
        obs = self._sanitize(obs)
        if hasattr(self, "critics"):
            return torch.cat([critic(obs) for critic in self.critics], dim=-1)
        return self.critic(obs)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def obs_segments(self):
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        self.std.copy_(self.std.clip(min=min, max=max))
