from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import instinct_rl.modules as modules
from instinct_rl.algorithms.wasabi import WasabiPPO
from instinct_rl.utils.utils import get_subobs_by_components


class OnnxParkourTeacher(nn.Module):
    """Small PyTorch mirror of the exported parkour teacher ONNX graphs."""

    def __init__(self, model_dir: str):
        super().__init__()
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ImportError("parkour teacher distillation requires the 'onnx' package in the training venv.") from exc

        depth_graph = onnx.load(f"{model_dir}/0-depth_encoder.onnx")
        actor_graph = onnx.load(f"{model_dir}/actor.onnx")
        depth_weights = {tensor.name: torch.from_numpy(numpy_helper.to_array(tensor).copy()) for tensor in depth_graph.graph.initializer}
        actor_weights = {tensor.name: torch.from_numpy(numpy_helper.to_array(tensor).copy()) for tensor in actor_graph.graph.initializer}

        self.depth_shape = (8, 18, 32)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(896, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )

        with torch.no_grad():
            self.depth_encoder[0].weight.copy_(depth_weights["conv.conv.0.weight"])
            self.depth_encoder[0].bias.copy_(depth_weights["conv.conv.0.bias"])
            for module_idx, prefix in ((3, "head.model.0"), (5, "head.model.2"), (7, "head.model.4")):
                self.depth_encoder[module_idx].weight.copy_(depth_weights[f"{prefix}.weight"])
                self.depth_encoder[module_idx].bias.copy_(depth_weights[f"{prefix}.bias"])
            for module_idx, prefix in ((0, "0"), (2, "2"), (4, "4"), (6, "6")):
                self.actor[module_idx].weight.copy_(actor_weights[f"{prefix}.weight"])
                self.actor[module_idx].bias.copy_(actor_weights[f"{prefix}.bias"])

        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, proprio_obs: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        depth_latent = self.depth_encoder(depth_image.reshape(-1, *self.depth_shape))
        return self.actor(torch.cat([proprio_obs, depth_latent], dim=-1))


class FlatPriorWasabiPPO(WasabiPPO):
    """WASABI/PPO with a frozen flat-locomotion policy used as a light action prior."""

    def __init__(
        self,
        *args,
        flat_prior_checkpoint: str | None = None,
        flat_prior_teacher_policy: dict | None = None,
        flat_prior_obs_components: list[str] | None = None,
        flat_prior_mask_component: str = "flat_prior_mask",
        flat_prior_expert_index: int = 0,
        flat_prior_kl_loss_coef: float = 0.0,
        flat_prior_expert_grad_scale: float = 1.0,
        flat_prior_gate_bias: float = 0.0,
        flat_prior_mask_source: str = "obs",
        flat_gate_prior_loss_coef: float = 0.0,
        flat_gate_target_prob: float = 0.75,
        flat_gate_nonflat_max_prob: float = 0.35,
        flat_gate_nonflat_loss_weight: float = 1.0,
        flat_gate_prior_warmup_iters: int = 0,
        flat_gate_prior_warmup_scale: float = 1.0,
        flat_prior_base_freeze_iters: int = 0,
        flat_prior_base_grad_scale_start: float | None = None,
        flat_prior_base_grad_scale: float = 1.0,
        flat_prior_base_grad_scale_schedule_iters: int = 0,
        residual_norm_penalty_coef: float = 0.0,
        residual_rate_penalty_coef: float = 0.0,
        residual_alpha_max_start: float | None = None,
        residual_alpha_max_end: float | None = None,
        residual_alpha_max_schedule_iters: int = 0,
        flat_prior_action_std_override: float | None = None,
        flat_prior_reset_action_std_on_load: bool = False,
        flat_prior_freeze_action_std: bool = False,
        flat_prior_action_std_freeze_iters: int = 0,
        flat_prior_action_std_max_start: float | None = None,
        flat_prior_action_std_max_end: float | None = None,
        flat_prior_action_std_max_schedule_iters: int = 0,
        flat_prior_kl_ignore_std: bool = False,
        residual_alpha_nonflat_target: float = 0.0,
        residual_alpha_nonflat_loss_coef: float = 0.0,
        residual_mixture_l2_loss_coef: float = 0.0,
        parkour_teacher_onnx_dir: str | None = None,
        parkour_teacher_obs_source: str = "critic_obs",
        parkour_teacher_obs_components: list[str] | None = None,
        parkour_teacher_depth_component: str = "depth_image",
        parkour_teacher_mask_component: str = "parkour_teacher_mask",
        parkour_teacher_loss_type: str = "huber",
        parkour_teacher_huber_delta: float = 0.15,
        parkour_teacher_imitation_loss_coef: float = 0.0,
        parkour_teacher_imitation_loss_coef_end: float | None = None,
        parkour_teacher_imitation_loss_hold_iters: int = 0,
        parkour_teacher_imitation_loss_schedule_iters: int = 0,
        parkour_teacher_cache_actions: bool = True,
        parkour_teacher_cache_batch_size: int = 16384,
        parkour_teacher_action_scale: list[float] | None = None,
        parkour_teacher_action_loss_weights: list[float] | None = None,
        parkour_teacher_action_clip: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flat_prior_checkpoint = flat_prior_checkpoint
        self.flat_prior_teacher_policy = flat_prior_teacher_policy or {
            "actor_hidden_dims": [256, 128, 128],
            "critic_hidden_dims": [256, 128, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
        }
        self.flat_prior_obs_components = flat_prior_obs_components or [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        self.flat_prior_mask_component = flat_prior_mask_component
        self.flat_prior_expert_index = flat_prior_expert_index
        self.flat_prior_kl_loss_coef = flat_prior_kl_loss_coef
        self.flat_prior_expert_grad_scale = flat_prior_expert_grad_scale
        self.flat_prior_gate_bias = flat_prior_gate_bias
        self.flat_prior_mask_source = flat_prior_mask_source
        self.flat_gate_prior_loss_coef = flat_gate_prior_loss_coef
        self.flat_gate_target_prob = flat_gate_target_prob
        self.flat_gate_nonflat_max_prob = flat_gate_nonflat_max_prob
        self.flat_gate_nonflat_loss_weight = flat_gate_nonflat_loss_weight
        self.flat_gate_prior_warmup_iters = flat_gate_prior_warmup_iters
        self.flat_gate_prior_warmup_scale = flat_gate_prior_warmup_scale
        self.flat_prior_base_freeze_iters = flat_prior_base_freeze_iters
        self.flat_prior_base_grad_scale_start = flat_prior_base_grad_scale_start
        self.flat_prior_base_grad_scale = flat_prior_base_grad_scale
        self.flat_prior_base_grad_scale_schedule_iters = flat_prior_base_grad_scale_schedule_iters
        self.residual_norm_penalty_coef = residual_norm_penalty_coef
        self.residual_rate_penalty_coef = residual_rate_penalty_coef
        self.residual_alpha_max_start = residual_alpha_max_start
        self.residual_alpha_max_end = residual_alpha_max_end
        self.residual_alpha_max_schedule_iters = residual_alpha_max_schedule_iters
        self.flat_prior_action_std_override = flat_prior_action_std_override
        self.flat_prior_reset_action_std_on_load = flat_prior_reset_action_std_on_load
        self.flat_prior_freeze_action_std = flat_prior_freeze_action_std
        self.flat_prior_action_std_freeze_iters = flat_prior_action_std_freeze_iters
        self.flat_prior_action_std_max_start = flat_prior_action_std_max_start
        self.flat_prior_action_std_max_end = flat_prior_action_std_max_end
        self.flat_prior_action_std_max_schedule_iters = flat_prior_action_std_max_schedule_iters
        self.flat_prior_kl_ignore_std = flat_prior_kl_ignore_std
        self.residual_alpha_nonflat_target = residual_alpha_nonflat_target
        self.residual_alpha_nonflat_loss_coef = residual_alpha_nonflat_loss_coef
        self.residual_mixture_l2_loss_coef = residual_mixture_l2_loss_coef
        self.parkour_teacher_onnx_dir = parkour_teacher_onnx_dir
        self.parkour_teacher_obs_source = parkour_teacher_obs_source
        self.parkour_teacher_obs_components = parkour_teacher_obs_components or [
            "teacher_base_ang_vel",
            "teacher_projected_gravity",
            "teacher_velocity_commands",
            "teacher_joint_pos",
            "teacher_joint_vel",
            "teacher_actions",
        ]
        self.parkour_teacher_depth_component = parkour_teacher_depth_component
        self.parkour_teacher_mask_component = parkour_teacher_mask_component
        self.parkour_teacher_loss_type = parkour_teacher_loss_type
        self.parkour_teacher_huber_delta = parkour_teacher_huber_delta
        self.parkour_teacher_imitation_loss_coef = parkour_teacher_imitation_loss_coef
        self._parkour_teacher_imitation_loss_coef_start = parkour_teacher_imitation_loss_coef
        self.parkour_teacher_imitation_loss_coef_end = parkour_teacher_imitation_loss_coef_end
        self.parkour_teacher_imitation_loss_hold_iters = parkour_teacher_imitation_loss_hold_iters
        self.parkour_teacher_imitation_loss_schedule_iters = parkour_teacher_imitation_loss_schedule_iters
        self.parkour_teacher_cache_actions = parkour_teacher_cache_actions
        self.parkour_teacher_cache_batch_size = parkour_teacher_cache_batch_size
        self.parkour_teacher_action_scale_values = parkour_teacher_action_scale
        self.parkour_teacher_action_loss_weight_values = parkour_teacher_action_loss_weights
        self.parkour_teacher_action_clip = parkour_teacher_action_clip
        self.parkour_teacher_action_scale = None
        self.parkour_teacher_action_loss_weights = None
        self.parkour_teacher_action_input_slice = None
        self.parkour_teacher_action_input_scale = None
        self.parkour_teacher_num_actions = 0
        self._parkour_teacher_action_cache: torch.Tensor | None = None
        self.flat_prior_enabled = bool(flat_prior_checkpoint)
        self.parkour_teacher_enabled = bool(parkour_teacher_onnx_dir)
        self._flat_prior_expert_params: list[nn.Parameter] = []
        self._prev_residual_action: torch.Tensor | None = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        if self.flat_prior_enabled:
            self._setup_flat_prior(obs_format, num_actions)
        if self.parkour_teacher_enabled:
            self._setup_parkour_teacher(obs_format, num_actions)
        self._apply_residual_alpha_schedule()

    def _setup_parkour_teacher(self, obs_format, num_actions):
        if num_actions != 29:
            raise RuntimeError(f"The exported parkour teacher outputs 29 actions, got student action dim {num_actions}.")
        self.parkour_teacher_num_actions = num_actions
        self.parkour_teacher = OnnxParkourTeacher(self.parkour_teacher_onnx_dir).to(self.device)
        segments = self._parkour_teacher_segments()
        missing = [
            name
            for name in self.parkour_teacher_obs_components + [self.parkour_teacher_depth_component]
            if name not in segments
        ]
        if missing:
            raise KeyError(f"Parkour teacher obs components missing from {self.parkour_teacher_obs_source}: {missing}")
        depth_shape = tuple(segments[self.parkour_teacher_depth_component])
        if depth_shape != self.parkour_teacher.depth_shape:
            raise RuntimeError(
                f"Parkour teacher depth shape mismatch: observation has {depth_shape}, "
                f"teacher expects {self.parkour_teacher.depth_shape}."
            )
        proprio_size = sum(int(torch.tensor(segments[name]).prod().item()) for name in self.parkour_teacher_obs_components)
        if proprio_size != 768:
            raise RuntimeError(f"Parkour teacher proprio size mismatch: expected 768, got {proprio_size}.")
        if self.parkour_teacher_action_scale_values is not None:
            if len(self.parkour_teacher_action_scale_values) != num_actions:
                raise RuntimeError(
                    f"Parkour teacher action scale length mismatch: expected {num_actions}, "
                    f"got {len(self.parkour_teacher_action_scale_values)}."
                )
            self.parkour_teacher_action_scale = torch.tensor(
                self.parkour_teacher_action_scale_values, dtype=torch.float32, device=self.device
            ).view(1, -1)
            action_component_offset = 0
            for name in self.parkour_teacher_obs_components:
                component_size = int(torch.tensor(segments[name]).prod().item())
                if name == "teacher_actions":
                    if component_size % num_actions != 0:
                        raise RuntimeError(
                            f"Parkour teacher action-history size mismatch: {component_size} is not "
                            f"divisible by action dim {num_actions}."
                        )
                    inverse_scale = (1.0 / self.parkour_teacher_action_scale).reshape(-1)
                    repeats = component_size // num_actions
                    self.parkour_teacher_action_input_slice = slice(
                        action_component_offset, action_component_offset + component_size
                    )
                    self.parkour_teacher_action_input_scale = inverse_scale.repeat(repeats).view(1, -1)
                    break
                action_component_offset += component_size
        if self.parkour_teacher_action_loss_weight_values is not None:
            if len(self.parkour_teacher_action_loss_weight_values) != num_actions:
                raise RuntimeError(
                    f"Parkour teacher action loss weight length mismatch: expected {num_actions}, "
                    f"got {len(self.parkour_teacher_action_loss_weight_values)}."
                )
            weights = torch.tensor(
                self.parkour_teacher_action_loss_weight_values, dtype=torch.float32, device=self.device
            ).view(1, -1)
            self.parkour_teacher_action_loss_weights = weights / weights.sum().clamp_min(1.0e-6)

    def _parkour_teacher_segments(self):
        if self.parkour_teacher_obs_source == "critic_obs":
            return self.actor_critic.critic_obs_segments
        if self.parkour_teacher_obs_source == "obs":
            return self.actor_critic.obs_segments
        raise ValueError(f"Unsupported parkour_teacher_obs_source {self.parkour_teacher_obs_source!r}.")

    def _parkour_teacher_obs_tensor(self, minibatch):
        return minibatch.critic_obs if self.parkour_teacher_obs_source == "critic_obs" else minibatch.obs

    def _parkour_teacher_loss_coef(self):
        end = self.parkour_teacher_imitation_loss_coef_end
        if end is None or self.parkour_teacher_imitation_loss_schedule_iters <= 0:
            return self._parkour_teacher_imitation_loss_coef_start
        schedule_iter = max(self.current_learning_iteration - self.parkour_teacher_imitation_loss_hold_iters, 0)
        progress = min(
            schedule_iter / self.parkour_teacher_imitation_loss_schedule_iters,
            1.0,
        )
        start = self._parkour_teacher_imitation_loss_coef_start
        return start + (end - start) * progress

    def _parkour_teacher_mask_from_minibatch(self, minibatch, fallback_flat_mask=None):
        obs = self._parkour_teacher_obs_tensor(minibatch)
        segments = self._parkour_teacher_segments()
        if self.parkour_teacher_mask_component in segments:
            mask = get_subobs_by_components(obs, [self.parkour_teacher_mask_component], segments)
            return mask.reshape(mask.shape[0], -1).max(dim=-1).values.clamp(0.0, 1.0)
        if fallback_flat_mask is not None:
            return 1.0 - fallback_flat_mask.reshape(-1).float()
        return torch.ones(obs.shape[0], device=obs.device)

    @torch.no_grad()
    def _parkour_teacher_actions_from_obs(self, obs, segments):
        proprio = get_subobs_by_components(obs, self.parkour_teacher_obs_components, segments)
        if self.parkour_teacher_action_input_slice is not None:
            proprio = proprio.clone()
            proprio[:, self.parkour_teacher_action_input_slice] *= self.parkour_teacher_action_input_scale
        depth = get_subobs_by_components(obs, [self.parkour_teacher_depth_component], segments, cat=False)[0]
        depth = depth.reshape(-1, *self.parkour_teacher.depth_shape)
        action = self.parkour_teacher(proprio, depth)
        if self.parkour_teacher_action_scale is not None:
            action = action * self.parkour_teacher_action_scale
        if self.parkour_teacher_action_clip is not None and self.parkour_teacher_action_clip > 0.0:
            action = action.clamp(-self.parkour_teacher_action_clip, self.parkour_teacher_action_clip)
        return action

    @torch.no_grad()
    def _parkour_teacher_actions_from_minibatch(self, minibatch):
        if (
            self._parkour_teacher_action_cache is not None
            and isinstance(minibatch.time_indices, torch.Tensor)
            and isinstance(minibatch.env_indices, torch.Tensor)
        ):
            return self._parkour_teacher_action_cache[minibatch.time_indices, minibatch.env_indices]
        obs = self._parkour_teacher_obs_tensor(minibatch)
        return self._parkour_teacher_actions_from_obs(obs, self._parkour_teacher_segments())

    @torch.no_grad()
    def _build_parkour_teacher_action_cache(self):
        if not self.parkour_teacher_enabled or not self.parkour_teacher_cache_actions:
            self._parkour_teacher_action_cache = None
            return
        storage_obs = (
            self.storage.critic_observations if self.parkour_teacher_obs_source == "critic_obs" else self.storage.observations
        )
        if storage_obs is None:
            self._parkour_teacher_action_cache = None
            return
        num_steps, num_envs = storage_obs.shape[:2]
        flat_obs = storage_obs.reshape(num_steps * num_envs, -1)
        flat_actions = torch.empty(num_steps * num_envs, self.parkour_teacher_num_actions, device=self.device)
        segments = self._parkour_teacher_segments()
        batch_size = max(int(self.parkour_teacher_cache_batch_size), 1)
        for start in range(0, flat_obs.shape[0], batch_size):
            end = min(start + batch_size, flat_obs.shape[0])
            flat_actions[start:end] = self._parkour_teacher_actions_from_obs(flat_obs[start:end], segments)
        self._parkour_teacher_action_cache = flat_actions.view(num_steps, num_envs, -1)

    def update(self, *args, **kwargs):
        self._build_parkour_teacher_action_cache()
        try:
            return super().update(*args, **kwargs)
        finally:
            self._parkour_teacher_action_cache = None

    def _setup_flat_prior(self, obs_format, num_actions):
        checkpoint = self._load_checkpoint(self.flat_prior_checkpoint)
        teacher_obs_format = self._build_teacher_obs_format(obs_format)
        teacher_policy_cfg = self.flat_prior_teacher_policy.copy()
        teacher_class_name = teacher_policy_cfg.pop("class_name", "ActorCritic")
        self.flat_prior_teacher = modules.build_actor_critic(
            teacher_class_name,
            teacher_policy_cfg,
            teacher_obs_format,
            num_actions=num_actions,
            num_rewards=1,
        ).to(self.device)
        self._load_teacher_actor(checkpoint)
        self.flat_prior_teacher.eval()
        for param in self.flat_prior_teacher.parameters():
            param.requires_grad_(False)

        self.flat_prior_normalizer = self._build_teacher_normalizer(checkpoint, teacher_obs_format)
        self._initialize_flat_expert(checkpoint, teacher_obs_format)
        self._bias_flat_gate()

    def _load_checkpoint(self, checkpoint_path: str) -> dict:
        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(checkpoint_path, map_location=self.device)

    def _apply_action_std_override(self):
        if self.flat_prior_action_std_override is None or not hasattr(self.actor_critic, "std"):
            return
        with torch.no_grad():
            self.actor_critic.std.fill_(self.flat_prior_action_std_override)

    def _action_std_max_cap(self):
        if self.flat_prior_action_std_max_start is None:
            return None
        start = self.flat_prior_action_std_max_start
        end = start if self.flat_prior_action_std_max_end is None else self.flat_prior_action_std_max_end
        if self.flat_prior_action_std_max_schedule_iters <= 0:
            return end
        progress = min(max(self.current_learning_iteration, 0) / self.flat_prior_action_std_max_schedule_iters, 1.0)
        return start + (end - start) * progress

    def _clamp_action_std(self, stats=None):
        if not hasattr(self.actor_critic, "std"):
            return
        cap = self._action_std_max_cap()
        if cap is None:
            return
        with torch.no_grad():
            self.actor_critic.std.clamp_(max=cap)
        if stats is not None:
            stats["flat_prior_action_std_max_cap"] = stats["flat_prior_action_std_max_cap"] + torch.tensor(
                cap, device=self.device
            )

    def _build_teacher_obs_format(self, obs_format):
        policy_segments = obs_format["policy"]
        missing = [name for name in self.flat_prior_obs_components if name not in policy_segments]
        if missing:
            raise KeyError(f"Flat-prior obs components missing from policy obs: {missing}")
        teacher_segments = OrderedDict((name, policy_segments[name]) for name in self.flat_prior_obs_components)
        return {"policy": teacher_segments}

    def _load_teacher_actor(self, checkpoint: dict):
        actor_state = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if key == "std" or key.startswith("actor.")
        }
        missing, unexpected = self.flat_prior_teacher.load_state_dict(actor_state, strict=False)
        unexpected = [key for key in unexpected if not key.startswith("critic.")]
        if unexpected:
            raise RuntimeError(f"Unexpected flat-prior teacher keys: {unexpected}")
        actor_missing = [key for key in missing if key == "std" or key.startswith("actor.")]
        if actor_missing:
            raise RuntimeError(f"Missing flat-prior teacher actor keys: {actor_missing}")

    def _build_teacher_normalizer(self, checkpoint: dict, teacher_obs_format):
        teacher_obs_size = sum(int(torch.tensor(shape).prod().item()) for shape in teacher_obs_format["policy"].values())
        normalizer = modules.build_normalizer(
            input_shape=teacher_obs_size,
            normalizer_class_name="EmpiricalNormalization",
            normalizer_kwargs={},
        ).to(self.device)
        if "policy_normalizer_state_dict" in checkpoint:
            normalizer.load_state_dict(checkpoint["policy_normalizer_state_dict"])
        normalizer.eval()
        return normalizer

    def _actor_input_segments(self):
        if hasattr(self.actor_critic, "encoders"):
            return self.actor_critic.encoders.output_segment
        return self.actor_critic.obs_segments

    def _component_offsets(self, segments):
        offsets = {}
        offset = 0
        for name, shape in segments.items():
            size = int(torch.tensor(shape).prod().item())
            offsets[name] = (offset, size)
            offset += size
        return offsets

    def _actor_core(self):
        actor = self.actor_critic.actor
        if isinstance(actor, nn.Sequential):
            return actor[0]
        return actor

    def _flat_prior_module(self):
        actor = self._actor_core()
        if hasattr(actor, "base"):
            return actor.base
        if hasattr(actor, "experts"):
            return actor.experts[self.flat_prior_expert_index]
        return None

    def _residual_alpha_cap(self):
        actor = self._actor_core()
        if not hasattr(actor, "alpha_max"):
            return None
        start = actor.alpha_max if self.residual_alpha_max_start is None else self.residual_alpha_max_start
        end = start if self.residual_alpha_max_end is None else self.residual_alpha_max_end
        if self.residual_alpha_max_schedule_iters <= 0:
            return end
        progress = min(max(self.current_learning_iteration, 0) / self.residual_alpha_max_schedule_iters, 1.0)
        return start + (end - start) * progress

    def _apply_residual_alpha_schedule(self, stats=None):
        alpha_cap = self._residual_alpha_cap()
        if alpha_cap is None:
            return
        actor = self._actor_core()
        actor.alpha_max = float(alpha_cap)
        if stats is not None:
            stats["residual_alpha_cap"] = torch.tensor(alpha_cap, device=self.device)

    def _flat_prior_base_grad_scale(self):
        if self.current_learning_iteration < self.flat_prior_base_freeze_iters:
            return 0.0
        target = self.flat_prior_base_grad_scale
        start = target if self.flat_prior_base_grad_scale_start is None else self.flat_prior_base_grad_scale_start
        if self.flat_prior_base_grad_scale_schedule_iters <= 0:
            scale = target
        else:
            ramp_iter = self.current_learning_iteration - self.flat_prior_base_freeze_iters
            progress = min(max(ramp_iter, 0) / self.flat_prior_base_grad_scale_schedule_iters, 1.0)
            scale = start + (target - start) * progress
        return scale * self.flat_prior_expert_grad_scale

    def act(self, obs, critic_obs):
        self._apply_residual_alpha_schedule()
        return super().act(obs, critic_obs)

    def _initialize_flat_expert(self, checkpoint: dict, teacher_obs_format):
        expert = self._flat_prior_module()
        if expert is None:
            raise TypeError("Flat-prior initialization requires an actor with a base module or experts ModuleList.")
        teacher_linears = [module for module in self.flat_prior_teacher.actor.modules() if isinstance(module, nn.Linear)]
        expert_linears = [module for module in expert.modules() if isinstance(module, nn.Linear)]
        if len(teacher_linears) != len(expert_linears):
            raise RuntimeError(
                f"Flat-prior teacher/expert layer count mismatch: {len(teacher_linears)} vs {len(expert_linears)}"
            )

        actor_segments = self._actor_input_segments()
        actor_offsets = self._component_offsets(actor_segments)
        normalizer_std = self.flat_prior_normalizer._std.squeeze(0) + self.flat_prior_normalizer.eps
        normalizer_mean = self.flat_prior_normalizer._mean.squeeze(0)

        with torch.no_grad():
            teacher_first = teacher_linears[0]
            expert_first = expert_linears[0]
            expert_first.weight.zero_()

            teacher_cursor = 0
            for name, shape in teacher_obs_format["policy"].items():
                size = int(torch.tensor(shape).prod().item())
                if name not in actor_offsets:
                    raise KeyError(f"Flat-prior component {name!r} not present after student encoders.")
                actor_offset, actor_size = actor_offsets[name]
                if actor_size != size:
                    raise RuntimeError(f"Flat-prior component {name!r} size mismatch: {actor_size} vs {size}")
                expert_first.weight[:, actor_offset : actor_offset + size].copy_(
                    teacher_first.weight[:, teacher_cursor : teacher_cursor + size]
                    / normalizer_std[teacher_cursor : teacher_cursor + size]
                )
                teacher_cursor += size

            if teacher_cursor != teacher_first.weight.shape[1]:
                raise RuntimeError(
                    f"Flat-prior teacher input size mismatch: copied {teacher_cursor}, "
                    f"checkpoint expects {teacher_first.weight.shape[1]}"
                )
            folded_bias = teacher_first.bias - torch.matmul(teacher_first.weight, normalizer_mean / normalizer_std)
            expert_first.bias.copy_(folded_bias)

            for teacher_layer, expert_layer in zip(teacher_linears[1:], expert_linears[1:]):
                if teacher_layer.weight.shape != expert_layer.weight.shape:
                    raise RuntimeError(
                        f"Flat-prior expert shape mismatch: {teacher_layer.weight.shape} vs {expert_layer.weight.shape}"
                    )
                expert_layer.weight.copy_(teacher_layer.weight)
                expert_layer.bias.copy_(teacher_layer.bias)
            if hasattr(self.actor_critic, "std") and "std" in checkpoint["model_state_dict"]:
                self.actor_critic.std.copy_(checkpoint["model_state_dict"]["std"])
            self._apply_action_std_override()
        self._flat_prior_expert_params = list(expert.parameters())

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.flat_prior_reset_action_std_on_load or self.flat_prior_freeze_action_std:
            self._apply_action_std_override()
        self._clamp_action_std()

    def _bias_flat_gate(self):
        if self.flat_prior_gate_bias == 0.0:
            return
        actor = self._actor_core()
        if hasattr(actor, "base"):
            return
        gate_linears = [module for module in actor.gate.modules() if isinstance(module, nn.Linear)]
        if not gate_linears:
            return
        with torch.no_grad():
            gate_linears[-1].bias[self.flat_prior_expert_index] += self.flat_prior_gate_bias

    def _teacher_obs_from_student_obs(self, obs):
        teacher_obs = get_subobs_by_components(obs, self.flat_prior_obs_components, self.actor_critic.obs_segments)
        return self.flat_prior_normalizer(teacher_obs)

    def _flat_prior_mask_from_obs(self, obs, obs_segments):
        if self.flat_prior_mask_component not in obs_segments:
            return torch.ones(obs.shape[0], device=obs.device)
        mask = get_subobs_by_components(obs, [self.flat_prior_mask_component], obs_segments)
        return mask.reshape(mask.shape[0], -1).max(dim=-1).values.clamp(0.0, 1.0)

    def _flat_prior_mask_from_minibatch(self, minibatch):
        if self.flat_prior_mask_source == "critic_obs":
            return self._flat_prior_mask_from_obs(minibatch.critic_obs, self.actor_critic.critic_obs_segments)
        return self._flat_prior_mask_from_obs(minibatch.obs, self.actor_critic.obs_segments)

    def _normal_kl(self, student_mean, student_std, teacher_mean, teacher_std):
        teacher_std = teacher_std.clamp_min(1e-6)
        student_std = student_std.clamp_min(1e-6)
        return torch.sum(
            torch.log(teacher_std / student_std)
            + (student_std.square() + (student_mean - teacher_mean).square()) / (2.0 * teacher_std.square())
            - 0.5,
            dim=-1,
        )

    def _mean_only_kl(self, student_mean, teacher_mean, teacher_std):
        teacher_std = teacher_std.clamp_min(1e-6)
        return torch.sum((student_mean - teacher_mean).square() / (2.0 * teacher_std.square()), dim=-1)

    def _masked_mean(self, values, mask):
        return torch.sum(values * mask) / torch.clamp(mask.sum(), min=1.0)

    def _actor_gate_scores(self):
        actor = self._actor_core()
        if hasattr(actor, "last_gate_scores"):
            return actor.last_gate_scores
        for module in actor.modules():
            if hasattr(module, "last_gate_scores"):
                return module.last_gate_scores
        return None

    def _flat_gate_prior_scale(self):
        if self.flat_gate_prior_warmup_iters <= 0:
            return 1.0
        if getattr(self, "current_learning_iteration", 0) < self.flat_gate_prior_warmup_iters:
            return self.flat_gate_prior_warmup_scale
        return 1.0

    def _compute_flat_gate_prior_loss(self, gate_scores, flat_mask, stats):
        flat_mask = flat_mask.reshape(-1).float()
        nonflat_mask = 1.0 - flat_mask
        flat_count = flat_mask.sum()
        nonflat_count = nonflat_mask.sum()
        expert_score = gate_scores[:, self.flat_prior_expert_index]

        gate_prior_loss = torch.zeros((), device=gate_scores.device)
        if flat_count > 0:
            flat_loss = torch.sum(torch.relu(self.flat_gate_target_prob - expert_score).square() * flat_mask) / flat_count
            gate_prior_loss = gate_prior_loss + flat_loss
            stats["flat_gate_prior_flat_loss"] = flat_loss.detach()
        if nonflat_count > 0:
            nonflat_loss = (
                torch.sum(torch.relu(expert_score - self.flat_gate_nonflat_max_prob).square() * nonflat_mask)
                / nonflat_count
            )
            gate_prior_loss = gate_prior_loss + self.flat_gate_nonflat_loss_weight * nonflat_loss
            stats["flat_gate_prior_nonflat_loss"] = nonflat_loss.detach()

        loss_scale = self._flat_gate_prior_scale()
        stats["flat_gate_prior_scale"] = torch.tensor(loss_scale, device=gate_scores.device)
        return gate_prior_loss * loss_scale

    def _record_actor_gate_stats(self, stats, flat_mask=None):
        gate_scores = self._actor_gate_scores()
        if gate_scores is None:
            return
        gate_scores = gate_scores.detach()
        top_expert = torch.argmax(gate_scores, dim=-1)
        gate_entropy = -(gate_scores * gate_scores.clamp_min(1e-8).log()).sum(dim=-1)
        stats["moe_actor_gate_entropy"] = gate_entropy.mean()
        for expert_idx in range(gate_scores.shape[-1]):
            expert_score = gate_scores[:, expert_idx]
            stats[f"moe_actor_gate_weight_expert_{expert_idx}"] = expert_score.mean()
            stats[f"moe_actor_gate_top1_expert_{expert_idx}"] = (top_expert == expert_idx).float().mean()
        if flat_mask is None:
            return
        flat_mask = flat_mask.detach().reshape(-1).float()
        flat_count = flat_mask.sum()
        nonflat_mask = 1.0 - flat_mask
        nonflat_count = nonflat_mask.sum()
        if flat_count > 0:
            stats["moe_actor_gate_flat_entropy"] = torch.sum(gate_entropy * flat_mask) / flat_count
            for expert_idx in range(gate_scores.shape[-1]):
                expert_score = gate_scores[:, expert_idx]
                stats[f"moe_actor_gate_flat_weight_expert_{expert_idx}"] = torch.sum(expert_score * flat_mask) / flat_count
                stats[f"moe_actor_gate_flat_top1_expert_{expert_idx}"] = (
                    torch.sum((top_expert == expert_idx).float() * flat_mask) / flat_count
                )
        if nonflat_count > 0:
            stats["moe_actor_gate_nonflat_entropy"] = torch.sum(gate_entropy * nonflat_mask) / nonflat_count
            for expert_idx in range(gate_scores.shape[-1]):
                expert_score = gate_scores[:, expert_idx]
                stats[f"moe_actor_gate_nonflat_weight_expert_{expert_idx}"] = (
                    torch.sum(expert_score * nonflat_mask) / nonflat_count
                )
                stats[f"moe_actor_gate_nonflat_top1_expert_{expert_idx}"] = (
                    torch.sum((top_expert == expert_idx).float() * nonflat_mask) / nonflat_count
                )

    def _record_residual_stats(self, stats, flat_mask=None):
        actor = self._actor_core()
        residual_action = getattr(actor, "last_residual_action", None)
        if residual_action is None:
            return
        residual_mixture = getattr(actor, "last_residual_mixture", residual_action)
        delta_h = getattr(actor, "last_delta_h", None)
        flat_hidden = getattr(actor, "last_flat_hidden", None)
        hidden = getattr(actor, "last_hidden", None)
        alpha = getattr(actor, "last_alpha", None)
        residual_action_norm = residual_action.detach().square().mean(dim=-1)
        residual_mixture_norm = residual_mixture.detach().square().mean(dim=-1)
        stats["residual_action_norm"] = residual_action_norm.mean()
        stats["residual_mixture_norm"] = residual_mixture_norm.mean()
        if delta_h is not None:
            delta_h_norm = delta_h.detach().square().mean(dim=-1)
            stats["residual_delta_h_norm"] = delta_h_norm.mean()
        if hidden is not None and flat_hidden is not None:
            hidden_shift_norm = (hidden.detach() - flat_hidden.detach()).square().mean(dim=-1)
            stats["residual_hidden_shift_norm"] = hidden_shift_norm.mean()
        projector = getattr(actor, "projector", None)
        if projector is not None:
            stats["residual_projector_weight_norm"] = projector.weight.detach().square().mean()
        if alpha is not None:
            alpha_value = alpha.detach().squeeze(-1)
            stats["residual_alpha_mean"] = alpha_value.mean()
            stats["residual_alpha_max"] = alpha_value.max()
        if flat_mask is None:
            return
        flat_mask = flat_mask.detach().reshape(-1).float()
        nonflat_mask = 1.0 - flat_mask
        if alpha is not None:
            stats["residual_alpha_flat_mean"] = self._masked_mean(alpha_value, flat_mask)
            stats["residual_alpha_nonflat_mean"] = self._masked_mean(alpha_value, nonflat_mask)
        stats["residual_action_flat_norm"] = self._masked_mean(residual_action_norm, flat_mask)
        stats["residual_action_nonflat_norm"] = self._masked_mean(residual_action_norm, nonflat_mask)
        stats["residual_mixture_flat_norm"] = self._masked_mean(residual_mixture_norm, flat_mask)
        stats["residual_mixture_nonflat_norm"] = self._masked_mean(residual_mixture_norm, nonflat_mask)
        if delta_h is not None:
            stats["residual_delta_h_flat_norm"] = self._masked_mean(delta_h_norm, flat_mask)
            stats["residual_delta_h_nonflat_norm"] = self._masked_mean(delta_h_norm, nonflat_mask)
        if hidden is not None and flat_hidden is not None:
            stats["residual_hidden_shift_flat_norm"] = self._masked_mean(hidden_shift_norm, flat_mask)
            stats["residual_hidden_shift_nonflat_norm"] = self._masked_mean(hidden_shift_norm, nonflat_mask)

    def compute_losses(self, minibatch):
        schedule_stats = {}
        self._apply_residual_alpha_schedule(schedule_stats)
        losses, inter_vars, stats = super().compute_losses(minibatch)
        stats.update(schedule_stats)
        flat_mask = None
        needs_flat_mask = (
            self.flat_prior_enabled
            and (
                self.flat_prior_kl_loss_coef > 0.0
                or self.flat_gate_prior_loss_coef > 0.0
                or (self.residual_alpha_nonflat_loss_coef > 0.0 and self.residual_alpha_nonflat_target > 0.0)
            )
        ) or self.parkour_teacher_enabled
        if needs_flat_mask:
            flat_mask = self._flat_prior_mask_from_minibatch(minibatch)
            stats["flat_prior_mask_fraction"] = flat_mask.detach().mean()
        if self.flat_prior_enabled and self.flat_prior_kl_loss_coef > 0.0:
            with torch.no_grad():
                teacher_obs = self._teacher_obs_from_student_obs(minibatch.obs)
                teacher_mean = self.flat_prior_teacher.act_inference(teacher_obs)
                teacher_std = self.flat_prior_teacher.std.unsqueeze(0).expand_as(teacher_mean)
            flat_mean_kl = self._mean_only_kl(
                self.actor_critic.action_mean,
                teacher_mean,
                teacher_std,
            )
            if self.flat_prior_kl_ignore_std:
                flat_kl = flat_mean_kl
            else:
                flat_kl = self._normal_kl(
                    self.actor_critic.action_mean,
                    self.actor_critic.action_std,
                    teacher_mean,
                    teacher_std,
                )
            masked_kl = self._masked_mean(flat_kl, flat_mask)
            losses["flat_prior_kl_loss"] = masked_kl
            stats["flat_prior_kl"] = masked_kl.detach()
            stats["flat_prior_mean_kl"] = self._masked_mean(flat_mean_kl, flat_mask).detach()
            if self.flat_prior_kl_ignore_std:
                flat_std_kl = self._normal_kl(teacher_mean, self.actor_critic.action_std, teacher_mean, teacher_std)
                stats["flat_prior_ignored_std_kl"] = self._masked_mean(flat_std_kl, flat_mask).detach()
        gate_scores = self._actor_gate_scores()
        if self.flat_prior_enabled and self.flat_gate_prior_loss_coef > 0.0 and gate_scores is not None:
            losses["flat_gate_prior_loss"] = self._compute_flat_gate_prior_loss(gate_scores, flat_mask, stats)
        actor = self._actor_core()
        alpha = getattr(actor, "last_alpha", None)
        residual_mixture = getattr(actor, "last_residual_mixture", None)
        if (
            self.residual_alpha_nonflat_loss_coef > 0.0
            and self.residual_alpha_nonflat_target > 0.0
            and alpha is not None
            and flat_mask is not None
        ):
            nonflat_mask = 1.0 - flat_mask
            alpha_shortfall = torch.clamp(self.residual_alpha_nonflat_target - alpha.squeeze(-1), min=0.0)
            alpha_shortfall = alpha_shortfall / max(self.residual_alpha_nonflat_target, 1.0e-6)
            losses["residual_alpha_nonflat_loss"] = self._masked_mean(alpha_shortfall.square(), nonflat_mask)
            stats["residual_alpha_nonflat_loss"] = losses["residual_alpha_nonflat_loss"].detach()
        if self.residual_mixture_l2_loss_coef > 0.0 and residual_mixture is not None:
            losses["residual_mixture_l2_loss"] = residual_mixture.square().mean()
            stats["residual_mixture_l2_loss"] = losses["residual_mixture_l2_loss"].detach()
        if self.parkour_teacher_enabled:
            teacher_loss_coef = self._parkour_teacher_loss_coef()
            self.parkour_teacher_imitation_loss_coef = teacher_loss_coef
            stats["parkour_teacher_imitation_loss_coef"] = torch.tensor(teacher_loss_coef, device=self.device)
            teacher_mask = self._parkour_teacher_mask_from_minibatch(minibatch, fallback_flat_mask=flat_mask)
            stats["parkour_teacher_mask_fraction"] = teacher_mask.detach().mean()
            with torch.no_grad():
                parkour_teacher_action = self._parkour_teacher_actions_from_minibatch(minibatch)
            stats["parkour_teacher_action_abs_max"] = parkour_teacher_action.detach().abs().max()
            if self.parkour_teacher_action_clip is not None and self.parkour_teacher_action_clip > 0.0:
                clip_threshold = self.parkour_teacher_action_clip - 1.0e-5
                stats["parkour_teacher_action_clip_fraction"] = (
                    parkour_teacher_action.detach().abs() >= clip_threshold
                ).float().mean()
            if self.parkour_teacher_loss_type == "huber":
                teacher_action_loss_by_dim = F.huber_loss(
                    self.actor_critic.action_mean,
                    parkour_teacher_action,
                    reduction="none",
                    delta=self.parkour_teacher_huber_delta,
                )
            elif self.parkour_teacher_loss_type == "mse":
                teacher_action_loss_by_dim = (self.actor_critic.action_mean - parkour_teacher_action).square()
            else:
                raise ValueError(f"Unsupported parkour_teacher_loss_type {self.parkour_teacher_loss_type!r}.")
            if self.parkour_teacher_action_loss_weights is not None:
                teacher_action_loss = (teacher_action_loss_by_dim * self.parkour_teacher_action_loss_weights).sum(dim=-1)
            else:
                teacher_action_loss = teacher_action_loss_by_dim.mean(dim=-1)
            losses["parkour_teacher_imitation_loss"] = self._masked_mean(teacher_action_loss, teacher_mask)
            stats["parkour_teacher_imitation_loss"] = losses["parkour_teacher_imitation_loss"].detach()
            teacher_delta_by_dim = (parkour_teacher_action - self.actor_critic.action_mean).detach().square()
            if self.parkour_teacher_action_loss_weights is not None:
                teacher_delta = (teacher_delta_by_dim * self.parkour_teacher_action_loss_weights).sum(dim=-1)
                stats["parkour_teacher_action_mse_unweighted"] = self._masked_mean(
                    teacher_delta_by_dim.mean(dim=-1), teacher_mask
                )
            else:
                teacher_delta = teacher_delta_by_dim.mean(dim=-1)
            stats["parkour_teacher_action_mse"] = self._masked_mean(teacher_delta, teacher_mask)
            action_groups = {
                "waist": [2, 5, 8],
                "arms": [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18, 21, 22],
                "legs": [11, 12, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28],
                "hips": [11, 12, 15, 16, 19, 20],
                "knees": [23, 24],
                "ankles": [25, 26, 27, 28],
            }
            for group_name, indices in action_groups.items():
                group_delta = teacher_delta_by_dim[:, indices].mean(dim=-1)
                stats[f"parkour_teacher_action_mse_{group_name}"] = self._masked_mean(group_delta, teacher_mask)
        self._record_actor_gate_stats(stats, flat_mask)
        self._record_residual_stats(stats, flat_mask)
        return losses, inter_vars, stats

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        if self._prev_residual_action is not None:
            done_mask = dones.to(self._prev_residual_action.device).bool().view(-1)
            self._prev_residual_action[done_mask] = 0.0

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rewards = super().compute_auxiliary_reward(obs_pack)
        actor = self._actor_core()
        residual_action = getattr(actor, "last_residual_action", None)
        if residual_action is None:
            return rewards
        residual_action = residual_action.detach()
        if self.residual_norm_penalty_coef != 0.0:
            rewards["residual_norm_penalty"] = -residual_action.square().mean(dim=-1, keepdim=True)
        if self.residual_rate_penalty_coef != 0.0:
            if self._prev_residual_action is None or self._prev_residual_action.shape != residual_action.shape:
                rate_penalty = torch.zeros(residual_action.shape[0], 1, device=residual_action.device)
            else:
                rate_penalty = -(residual_action - self._prev_residual_action).square().mean(dim=-1, keepdim=True)
            rewards["residual_rate_penalty"] = rate_penalty
        self._prev_residual_action = residual_action.clone()
        return rewards

    def gradient_step(self, loss: torch.Tensor, average_stats: dict):
        self.optimizer.zero_grad()
        loss.backward()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.actor_critic.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        base_param_backups = []
        if self.flat_prior_enabled and self._flat_prior_expert_params:
            grad_scale = self._flat_prior_base_grad_scale()
            freeze_base = grad_scale == 0.0
            if grad_scale < 1.0:
                base_param_backups = [
                    (param, param.detach().clone()) for param in self._flat_prior_expert_params if param.requires_grad
                ]
            if freeze_base:
                average_stats["flat_prior_base_frozen"] = average_stats["flat_prior_base_frozen"] + torch.tensor(
                    1.0, device=self.device
                )
            for param in self._flat_prior_expert_params:
                if param.grad is not None:
                    param.grad.mul_(grad_scale)
            average_stats["flat_prior_base_grad_scale"] = average_stats["flat_prior_base_grad_scale"] + torch.tensor(
                grad_scale, device=self.device
            )
        freeze_std = self.flat_prior_freeze_action_std or (
            self.flat_prior_action_std_freeze_iters > 0
            and self.current_learning_iteration < self.flat_prior_action_std_freeze_iters
        )
        if freeze_std and hasattr(self.actor_critic, "std"):
            std_backup = self.actor_critic.std.detach().clone()
            if self.actor_critic.std.grad is not None:
                self.actor_critic.std.grad.zero_()
            average_stats["flat_prior_action_std_frozen"] = average_stats[
                "flat_prior_action_std_frozen"
            ] + torch.tensor(1.0, device=self.device)
        else:
            std_backup = None
        grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        average_stats["grad_norm"] = average_stats["grad_norm"] + grad_norm.detach()
        self.optimizer.step()
        for param, backup in base_param_backups:
            param.data.copy_(backup + (param.data - backup) * grad_scale)
        if std_backup is not None:
            self.actor_critic.std.data.copy_(std_backup)
        self._clamp_action_std(average_stats)
