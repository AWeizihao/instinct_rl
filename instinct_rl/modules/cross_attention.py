import numpy as np
import torch
import torch.nn as nn

from .mlp import MlpModel


class _CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, activation):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            getattr(nn, activation)() if hasattr(nn, activation) else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, query, context):
        attended, _ = self.attn(query, context, context, need_weights=False)
        query = self.norm_query(query + self.dropout(attended))
        query = self.norm_ff(query + self.dropout(self.ff(query)))
        return query


class DepthProprioCrossAttentionModel(nn.Module):
    """Depth patch tokens attended by a proprioceptive/command query."""

    def __init__(
        self,
        input_shapes,
        output_size,
        depth_component_index=0,
        patch_size=(3, 4),
        num_heads=4,
        num_layers=1,
        d_model=128,
        dim_feedforward=256,
        dropout=0.02,
        activation="GELU",
        nonlinearity="ReLU",
        context_hidden_sizes=(256,),
        output_hidden_sizes=(128,),
    ):
        super().__init__()
        self.input_shapes = [tuple(shape) for shape in input_shapes]
        self.output_size = output_size
        self.depth_component_index = depth_component_index
        self.component_sizes = [int(np.prod(shape)) for shape in self.input_shapes]
        self.depth_shape = self.input_shapes[depth_component_index]
        if len(self.depth_shape) != 3:
            raise ValueError(f"DepthProprioCrossAttentionModel expects depth shape (C,H,W), got {self.depth_shape}.")
        depth_channels, depth_height, depth_width = self.depth_shape
        patch_h, patch_w = patch_size
        if depth_height % patch_h != 0 or depth_width % patch_w != 0:
            raise ValueError(f"patch_size {patch_size} must divide depth image shape {self.depth_shape}.")
        context_size = sum(size for idx, size in enumerate(self.component_sizes) if idx != depth_component_index)
        if context_size <= 0:
            raise ValueError("DepthProprioCrossAttentionModel needs at least one non-depth context component.")

        self.patch_embed = nn.Conv2d(depth_channels, d_model, kernel_size=patch_size, stride=patch_size)
        num_tokens = (depth_height // patch_h) * (depth_width // patch_w)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.context_layer = MlpModel(
            input_size=context_size,
            hidden_sizes=list(context_hidden_sizes) + [d_model],
            output_size=None,
            nonlinearity=nonlinearity,
        )
        self.cross_layers = nn.ModuleList(
            [
                _CrossAttentionBlock(d_model, num_heads, dim_feedforward, dropout, activation)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = MlpModel(
            input_size=d_model,
            hidden_sizes=list(output_hidden_sizes) + [output_size],
            output_size=None,
            nonlinearity=nonlinearity,
        )

    def forward(self, x):
        chunks = torch.split(x, self.component_sizes, dim=-1)
        depth = chunks[self.depth_component_index].reshape(*x.shape[:-1], *self.depth_shape)
        context = torch.cat([chunk for idx, chunk in enumerate(chunks) if idx != self.depth_component_index], dim=-1)
        leading_shape = depth.shape[:-3]
        depth = depth.reshape(-1, *self.depth_shape)
        context = context.reshape(-1, context.shape[-1])

        tokens = self.patch_embed(depth).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        query = self.context_layer(context).unsqueeze(1)
        for layer in self.cross_layers:
            query = layer(query, tokens)
        output = self.output_layer(query.squeeze(1))
        return output.reshape(*leading_shape, self.output_size)
