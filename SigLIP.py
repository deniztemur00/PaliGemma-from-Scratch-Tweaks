from typing import Optional, Tuple
import torch
from torch import nn
from configs import SigLIPVisionConfig


class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_embeds = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, x):
        _, _, H, W = x.shape
        patches = self.image_embeds(x)
        embeds = patches.flatten(2).transpose(1, 2)
        embeds = embeds + self.position_embeddings(self.position_ids)
        ## [batch_size, num_patches, embed_dim]
        return embeds


class SigLIPSelfAttention(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = query.view(
            batch_size, num_patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, num_patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, num_patches, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # flash attention
        attn = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout=self.attn_dropout,
            scale=self.scaling,
        )

        # attn_weights = (query @ key.transpose(-2, -1)) * self.scaling

        # assert attn_weights.size() == (batch_size, self.num_heads, num_patches, num_patches)
        #
        # attn_weights = attn_weights.softmax(dim=-1)
        # attn_weights = self.attn_dropout(attn_weights)
        # attn = attn_weights @ value

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.reshape(batch_size, num_patches, self.embed_dim)
        attn = self.projection(attn)

        return attn


class SigLIPMLP(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SigLIPVisionEncoderLayer(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.self_attn = SigLIPSelfAttention(config)
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = x + residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class SigLIPVisionEncoder(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SigLIPVisionTransformer(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        return x


class SigLIPVisionModel(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLIPVisionTransformer(config)

    def forward(self, x) -> Tuple:
        return self.vision_model(x)
