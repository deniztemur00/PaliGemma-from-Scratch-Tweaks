from typing import Optional
import torch.nn as nn


class SigLIPVisionConfig(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        self.num_tokens = num_image_tokens


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=512,
        rms_norm_epsilon=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        hidden_dropout_prob=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_epsilon = rms_norm_epsilon
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = hidden_dropout_prob
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False


class PaliGemmaConfig:
    def __init__(
        self,
        text_config: None,
        vision_config: None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        project_dim=2048,
        hidden_size=2048,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.project_dim = project_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False
