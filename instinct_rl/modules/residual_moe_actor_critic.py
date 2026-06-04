import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCritic, get_activation
from .encoder_actor_critic import EncoderActorCriticMixin
from .moe import MoeLayer


class FlatTrunkHead(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dims, activation):
        super().__init__()
        if not hidden_dims:
            raise ValueError("FlatTrunkHead requires at least one hidden layer for latent residuals.")
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(get_activation(activation))
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, num_actions)
        self.hidden_dim = last_dim

    def forward_hidden(self, x):
        return self.trunk(x)

    def forward_head(self, h):
        return self.head(h)

    def forward(self, x):
        return self.forward_head(self.forward_hidden(x))


class ResidualMoEActor(nn.Module):
    def __init__(
        self,
        input_dim,
        num_actions,
        num_residual_experts=3,
        base_hidden_dims=(256, 128, 128),
        residual_hidden_dims=(256, 128, 64),
        gate_hidden_dims=(128,),
        alpha_hidden_dims=(128, 64),
        activation="elu",
        residual_delta_scale=0.5,
        latent_residual_dim=8,
        residual_mode="latent",
        residual_projector_init_std=1.0e-3,
        alpha_max=1.0,
        alpha_fixed_value=-1.0,
        alpha_init_bias=-2.0,
        residual_output_init_std=1.0e-3,
    ):
        super().__init__()
        self.num_residual_experts = num_residual_experts
        self.residual_delta_scale = residual_delta_scale
        self.latent_residual_dim = latent_residual_dim
        self.residual_mode = residual_mode
        self.alpha_max = alpha_max
        self.alpha_fixed_value = alpha_fixed_value
        self.act_fn = get_activation(activation)
        self.base = FlatTrunkHead(input_dim, num_actions, base_hidden_dims, activation)
        residual_output_dim = num_actions if residual_mode == "action" else latent_residual_dim
        if residual_mode not in ("latent", "action"):
            raise ValueError(f"Unsupported residual_mode {residual_mode!r}; expected 'latent' or 'action'.")
        self.experts = nn.ModuleList(
            [
                self._build_mlp(input_dim, residual_output_dim, residual_hidden_dims, activation)
                for _ in range(num_residual_experts)
            ]
        )
        self.projector = nn.Linear(latent_residual_dim, self.base.hidden_dim) if residual_mode == "latent" else None
        self.gate = self._build_mlp(input_dim, num_residual_experts, gate_hidden_dims, activation)
        self.alpha = self._build_mlp(input_dim, 1, alpha_hidden_dims, activation)
        self._init_residual_outputs(residual_output_init_std)
        if self.projector is not None:
            self._init_projector(residual_projector_init_std)
        self._init_alpha(alpha_init_bias)

    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation):
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(get_activation(activation))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_residual_outputs(self, init_std):
        for expert in self.experts:
            last_linear = [module for module in expert.modules() if isinstance(module, nn.Linear)][-1]
            nn.init.normal_(last_linear.weight, mean=0.0, std=init_std)
            nn.init.zeros_(last_linear.bias)

    def _init_alpha(self, init_bias):
        last_linear = [module for module in self.alpha.modules() if isinstance(module, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, init_bias)

    def _init_projector(self, init_std):
        nn.init.normal_(self.projector.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        h_flat = self.base.forward_hidden(x)
        base_action = self.base.forward_head(h_flat)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        if self.alpha_fixed_value >= 0.0:
            alpha = torch.full((x.shape[0], 1), self.alpha_fixed_value, device=x.device, dtype=x.dtype)
        else:
            alpha = self.alpha_max * torch.sigmoid(self.alpha(x))
        expert_outputs = torch.stack([torch.tanh(expert(x)) for expert in self.experts], dim=1)
        residual_mixture = torch.einsum("be,bea->ba", gate_scores, expert_outputs)
        if self.residual_mode == "action":
            delta_h = None
            h = h_flat
            residual_action = alpha * self.residual_delta_scale * residual_mixture
            action = base_action + residual_action
        else:
            delta_h = self.projector(residual_mixture)
            h = h_flat + alpha * self.residual_delta_scale * delta_h
            action = self.base.forward_head(h)
            residual_action = action - base_action
        self.last_base_action = base_action
        self.last_gate_scores = gate_scores
        self.last_alpha = alpha
        self.last_expert_outputs = expert_outputs
        self.last_residual_mixture = residual_mixture
        self.last_delta_z = residual_mixture
        self.last_delta_h = delta_h
        self.last_hidden = h
        self.last_flat_hidden = h_flat
        self.last_residual_action = residual_action
        return action


class ResidualMoEActorCritic(ActorCritic):
    def __init__(
        self,
        obs_format,
        num_actions,
        actor_hidden_dims=(256, 128, 128),
        residual_hidden_dims=(256, 128, 64),
        gate_hidden_dims=(128,),
        alpha_hidden_dims=(128, 64),
        critic_hidden_dims=(256, 128, 64),
        activation="elu",
        init_noise_std=1.0,
        num_rewards=1,
        mu_activation=None,
        num_residual_experts=3,
        residual_delta_scale=0.5,
        latent_residual_dim=8,
        residual_mode="latent",
        residual_projector_init_std=1.0e-3,
        alpha_max=1.0,
        alpha_fixed_value=-1.0,
        alpha_init_bias=-2.0,
        residual_output_init_std=1.0e-3,
        **kwargs,
    ):
        self.num_residual_experts = num_residual_experts
        self.residual_hidden_dims = residual_hidden_dims
        self.gate_hidden_dims = gate_hidden_dims
        self.alpha_hidden_dims = alpha_hidden_dims
        self.residual_delta_scale = residual_delta_scale
        self.latent_residual_dim = latent_residual_dim
        self.residual_mode = residual_mode
        self.residual_projector_init_std = residual_projector_init_std
        self.alpha_max = alpha_max
        self.alpha_fixed_value = alpha_fixed_value
        self.alpha_init_bias = alpha_init_bias
        self.residual_output_init_std = residual_output_init_std
        super().__init__(
            obs_format,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            num_rewards,
            mu_activation,
            **kwargs,
        )

    def _build_actor(self, num_actions):
        actor = ResidualMoEActor(
            self.mlp_input_dim_a,
            num_actions,
            num_residual_experts=self.num_residual_experts,
            base_hidden_dims=self.actor_hidden_dims,
            residual_hidden_dims=self.residual_hidden_dims,
            gate_hidden_dims=self.gate_hidden_dims,
            alpha_hidden_dims=self.alpha_hidden_dims,
            activation=self.activation,
            residual_delta_scale=self.residual_delta_scale,
            latent_residual_dim=self.latent_residual_dim,
            residual_mode=self.residual_mode,
            residual_projector_init_std=self.residual_projector_init_std,
            alpha_max=self.alpha_max,
            alpha_fixed_value=self.alpha_fixed_value,
            alpha_init_bias=self.alpha_init_bias,
            residual_output_init_std=self.residual_output_init_std,
        )
        if self.mu_activation:
            return nn.Sequential(actor, get_activation(self.mu_activation))
        return actor

    def _build_critic(self, num_values=1):
        return MoeLayer(
            self.mlp_input_dim_c,
            self.num_residual_experts + 1,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=[],
        )


class EncoderResidualMoEActorCritic(EncoderActorCriticMixin, ResidualMoEActorCritic):
    pass
