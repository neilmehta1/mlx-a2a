from dataclasses import dataclass
from typing import Dict, Optional, Union

import mlx.nn as nn
import mlx.core as mx

from mlx_lm.models.base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 4096

    # talker args??
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    # Thinker specific parameters
    thinker_hidden_size: int = 4096
    thinker_num_hidden_layers: int = 36
    thinker_num_attention_heads: int = 32
    thinker_intermediate_size: int = 11008
    thinker_num_key_value_heads: int = 32

    # Audio tower parameters
    audio_tower_layers: int = 32

    # Visual parameters
    visual_blocks: int = 32

    # NM: cleanup
    talker_head_dim = 64
    out_hidden_size: int = 2048
    spatial_merge_size: int = 2
    vision_hidden_size: int = 1280
    dit_config_depth: int = 22
    look_ahead_layers = ([10],)
    look_backward_layers = [0, 20]
    dit_dropout = 0.1
    dit_ff_mult = 2


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, bias: bool = True):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            if k_cache is not None:
                k = mx.concatenate([k_cache, k], axis=2)
                v = mx.concatenate([v_cache, v], axis=2)
            cache = (k, v)

        # Scaled dot-product attention
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn, v)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), cache


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, bias: bool = True):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = Attention(args, bias=bias)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = MLP(args)

    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.input_layernorm(x)
        attn_output, cache = self.self_attn(x, mask, cache)
        x = residual + attn_output

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x, cache


class Qwen2_5_TalkerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, bias=True) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h, cache[i] = layer(h, mask, layer_cache)

        return self.norm(h), cache


class Qwen2_5_Talker(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Qwen2_5_TalkerModel(args)
        self.codec_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.thinker_to_talker_proj = nn.Linear(
            args.thinker_hidden_size, args.hidden_size, bias=True
        )

    def __call__(self, inputs, mask=None, cache=None, thinker_output=None):
        h, cache = self.model(inputs, mask, cache)
        return h, cache


class AudioTower(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Simplified implementation of audio tower
        self.layers = [
            nn.Linear(args.hidden_size, args.hidden_size)
            for _ in range(args.audio_tower_layers)
        ]
        self.ln_post = nn.LayerNorm(args.hidden_size)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.audio_bos_eos_token = nn.Embedding(2, args.hidden_size)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.ln_post(x)
        return self.proj(x)


class VisualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.norm2 = nn.LayerNorm(args.hidden_size)
        self.attn = nn.Module()
        self.attn.q = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.attn.k = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.attn.v = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.attn.proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=True
        )
        self.mlp.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=True
        )
        self.mlp.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=True
        )

    def __call__(self, x):
        # Simplified implementation
        return x


class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]


class Visual(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.blocks = [VisualBlock(args) for _ in range(args.visual_blocks)]
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(
            args.hidden_size, args.hidden_size, bias=False
        )

        self.merger = Qwen2_5OmniPatchMerger(
            dim=args.out_hidden_size,
            context_dim=args.vision_hidden_size,
            spatial_merge_size=args.spatial_merge_size,
        )

    def __call__(self, x):
        # Simplified implementation
        return x


class Qwen2_5_ThinkerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, bias=True)
            for _ in range(args.thinker_num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        return inputs


class Qwen2_5_Thinker(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Qwen2_5_ThinkerModel(args)
        self.audio_tower = AudioTower(args)
        self.visual = Visual(args)
        # self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs, mask=None, cache=None, audio_input=None, visual_input=None
    ):
        # Simplified implementation
        h, cache = self.model(inputs, mask, cache)
        return h, cache


class Code2WavBigVGANModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Simplified implementation
        self.conv_pre = nn.Linear(1, 1, bias=True)
        self.conv_post = nn.Linear(1, 1, bias=False)
        self.activation_post = nn.Module()
        self.activation_post.act = nn.Module()
        self.activation_post.act.alpha = mx.array([1.0])
        self.activation_post.act.beta = mx.array([1.0])

        self.resblocks = []
        for i in range(18):  # Based on weight mapping
            block = nn.Module()
            block.activations = []
            for j in range(6):  # Based on weight mapping
                act = nn.Module()
                act.act = nn.Module()
                act.act.alpha = mx.array([1.0])
                act.act.beta = mx.array([1.0])
                block.activations.append(act)

            block.convs1 = []
            block.convs2 = []
            for j in range(3):  # Based on weight mapping
                block.convs1.append(nn.Linear(1, 1, bias=True))
                block.convs2.append(nn.Linear(1, 1, bias=True))

            self.resblocks.append(block)

        self.ups = []
        for i in range(6):  # Based on weight mapping
            up = []
            up.append(nn.Linear(1, 1, bias=True))
            self.ups.append(up)

    def __call__(self, x):
        # Simplified implementation
        return x


class DiTMLP(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = [
            nn.Linear(dim, inner_dim),
            nn.GELU(approx="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        ]


class DiTAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.config = args
        self.dim = args.hidden_size
        self.heads = args.num_attention_heads
        self.inner_dim = args.talker_head_dim * args.num_attention_heads
        self.dropout = args.dit_dropout
        # self._attn_implementation = args._attn_implementation
        self.is_causal = False

        self.to_q = nn.Linear(args.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(args.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(args.hidden_size, self.inner_dim)

        self.to_out = [
            nn.Linear(self.inner_dim, args.hidden_size),
            nn.Dropout(args.dit_dropout),
        ]


class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)


class DiTDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = Qwen2_5_OmniAdaLayerNormZero(args.hidden_size)

        self.attn = DiTAttention(args)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(args.hidden_size, affine=False, eps=1e-6)
        self.ff = DiTMLP(
            dim=args.hidden_size, mult=args.dit_ff_mult, dropout=args.dit_dropout
        )


# time step conditioning embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = [nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)]


class Code2WavDITModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Simplified implementation
        self.input_embed = nn.Module()
        self.input_embed.proj = nn.Linear(1, 1, bias=True)
        self.input_embed.spk_encoder = nn.Module()

        self.text_embed = nn.Module()
        self.text_embed.codec_embed = nn.Embedding(1, 1)

        self.time_embed = DiTTimestepEmbedding(args.hidden_size)

        self.rotary_embed = nn.Module()
        self.rotary_embed.inv_freq = mx.array([1.0])

        self.transformer_blocks = []
        for i in range(args.num_hidden_layers):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    args,
                    look_ahead_block=1 if i in args.look_ahead_layers else 0,
                    look_backward_block=1 if i in args.look_backward_layers else 0,
                )
            )

        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(1, 1, bias=True)

        self.proj_out = nn.Linear(1, 1, bias=True)

    def __call__(self, x):
        # Simplified implementation
        return x


class Qwen2_5_Token2wav(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.code2wav_bigvgan_model = Code2WavBigVGANModel(args)
        self.code2wav_dit_model = Code2WavDITModel(args)

    def __call__(self, x):
        # Simplified implementation
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.talker = Qwen2_5_Talker(args)
        self.thinker = Qwen2_5_Thinker(args)
        self.token2wav = Qwen2_5_Token2wav(args)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        # Simplified implementation - would need to be expanded based on actual usage
        h, cache = self.talker(inputs, mask, cache)
        return h

    def sanitize(self, weights):
        # Remove any weights that shouldn't be loaded
        return weights

    @property
    def layers(self):
        return self.talker.model.layers
