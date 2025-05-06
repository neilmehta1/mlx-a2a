import inspect
from dataclasses import dataclass, field
import math
from typing import Dict, Union, List, Any

import mlx.nn as nn
import mlx.core as mx

from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from torch import kaiser_window as torch_kaiser_window
from torch import float32 as torch_kaiser_window_dtype
from torch import sinc as torch_sinc
from torch import arange as torch_arange


@dataclass
class ThinkerTextConfigArgs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: Dict[str, Any]
    attention_dropout: float


@dataclass
class ThinkerAudioConfigArgs:
    d_model: int
    encoder_attention_heads: int
    attention_dropout: float
    activation_dropout: float
    encoder_ffn_dim: int
    encoder_layers: int
    num_mel_bins: int
    max_source_positions: int
    scale_embedding: bool
    n_window: int
    output_dim: int
    dropout: float


@dataclass
class ThinkerVisionConfigArgs:
    depth: int
    hidden_size: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    in_channels: int
    out_hidden_size: int
    spatial_merge_size: int


@dataclass
class ThinkerConfigArgs:
    text_config: ThinkerTextConfigArgs
    audio_config: ThinkerAudioConfigArgs
    vision_config: ThinkerVisionConfigArgs

    @classmethod
    def from_dict(cls, params: dict):
        text_params = params.get("text_config", {})
        audio_params = params.get("audio_config", {})
        vision_params = params.get("vision_config", {})

        text_cfg = ThinkerTextConfigArgs(
            **{
                k: v
                for k, v in text_params.items()
                if k in inspect.signature(ThinkerTextConfigArgs).parameters
            }
        )
        audio_cfg = ThinkerAudioConfigArgs(
            **{
                k: v
                for k, v in audio_params.items()
                if k in inspect.signature(ThinkerAudioConfigArgs).parameters
            }
        )
        vision_cfg = ThinkerVisionConfigArgs(
            **{
                k: v
                for k, v in vision_params.items()
                if k in inspect.signature(ThinkerVisionConfigArgs).parameters
            }
        )

        other_params = {
            k: v
            for k, v in params.items()
            if k in inspect.signature(cls).parameters
            and k not in ["text_config", "audio_config", "vision_config"]
        }
        return cls(
            text_config=text_cfg,
            audio_config=audio_cfg,
            vision_config=vision_cfg,
            **other_params,
        )


@dataclass
class TalkerConfigArgs:
    vocab_size: int
    embedding_size: int
    num_hidden_layers: int
    hidden_size: int
    head_dim: int
    num_key_value_heads: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: Dict[str, Any]
    attention_dropout: float

    @classmethod
    def from_dict(cls, params: dict):
        # Filter params to only include keys defined in this dataclass
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class Token2WavDitConfigArgs:
    ff_mult: int  # Corresponds to dit_ff_mult
    dropout: float  # Corresponds to dit_dropout
    dim: int  # Corresponds to dit_hidden_size
    heads: int  # Corresponds to dit_num_attention_heads
    head_dim: int
    num_embeds: int
    emb_dim: int
    repeats: int
    depth: int
    enc_dim: int
    enc_emb_dim: int
    mel_dim: int
    enc_channels: List[int]
    enc_kernel_sizes: List[int]
    enc_dilations: List[int]
    enc_res2net_scale: int
    enc_se_channels: int
    enc_attention_channels: int
    look_ahead_layers: List[int] = field(default_factory=lambda: [10])
    look_backward_layers: List[int] = field(default_factory=lambda: [0, 20])


@dataclass
class Token2WavBigVGANConfigArgs:
    mel_dim: int
    upsample_initial_channel: int
    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]


@dataclass
class Token2WavConfigArgs:
    dit_config: Token2WavDitConfigArgs
    bigvgan_config: Token2WavBigVGANConfigArgs

    @classmethod
    def from_dict(cls, params: dict):
        dit_params = params.get("dit_config", {})
        bigvgan_params = params.get("bigvgan_config", {})

        dit_cfg = Token2WavDitConfigArgs(
            **{
                k: v
                for k, v in dit_params.items()
                if k in inspect.signature(Token2WavDitConfigArgs).parameters
            }
        )
        bigvgan_cfg = Token2WavBigVGANConfigArgs(
            **{
                k: v
                for k, v in bigvgan_params.items()
                if k in inspect.signature(Token2WavBigVGANConfigArgs).parameters
            }
        )

        other_params = {
            k: v
            for k, v in params.items()
            if k in inspect.signature(cls).parameters
            and k not in ["dit_config", "bigvgan_config"]
        }
        return cls(dit_config=dit_cfg, bigvgan_config=bigvgan_cfg, **other_params)


# --- Main ModelArgs ---
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    torch_dtype: str
    thinker_config: dict = field(default_factory=dict)
    talker_config: dict = field(default_factory=dict)
    token2wav_config: dict = field(default_factory=dict)

    thinker: ThinkerConfigArgs = field(init=False)
    talker: TalkerConfigArgs = field(init=False)
    token2wav: Token2WavConfigArgs = field(init=False)

    def __post_init__(self):
        self.thinker = ThinkerConfigArgs.from_dict(self.thinker_config or {})
        self.talker = TalkerConfigArgs.from_dict(self.talker_config or {})
        self.token2wav = Token2WavConfigArgs.from_dict(self.token2wav_config or {})


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        args: Union[TalkerConfigArgs, ThinkerConfigArgs],
        *,
        bias: bool = False,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()


class Qwen2_5OmniRotaryEmbedding(nn.Module):
    def __init__(
        self,
        args: Union[TalkerConfigArgs, ThinkerConfigArgs],
        head_dim: int,
        device=None,
    ):
        super().__init__()

        # Determine rope_type and max_position_embeddings based on the type of args
        if isinstance(args, TalkerConfigArgs):
            rope_type = (
                args.rope_scaling.get("rope_type", "default")
                if args.rope_scaling
                else "default"
            )
            max_position_embeddings = args.max_position_embeddings
            base = args.rope_theta
        elif isinstance(args, ThinkerConfigArgs):
            rope_type = (
                args.text_config.rope_scaling.get("rope_type", "default")
                if args.text_config.rope_scaling
                else "default"
            )
            max_position_embeddings = args.text_config.max_position_embeddings
            base = args.text_config.rope_theta
        else:
            raise NotImplementedError

        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        # rope init
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.attention_scaling = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        self._inv_freq = 1.0 / (
            base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim)
        )


class Qwen2_5OmniAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        args: Union[TalkerConfigArgs, ThinkerConfigArgs],
        *,
        hidden_size: int,
        layer_idx: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise Exception(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # NM: fix this
        attention_dropout = 0.0
        rope_scaling = {
            "mrope_section": [16, 16, 0],
            "rope_type": "default",
            "type": "default",
        }

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        # self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(args=args, head_dim=head_dim)


class Qwen2_5OmniDecoderLayer(nn.Module):
    def __init__(
        self,
        args: Union[TalkerConfigArgs, ThinkerConfigArgs],
        *,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()

        if isinstance(args, TalkerConfigArgs):
            rms_norm_eps = args.rms_norm_eps
        elif isinstance(args, ThinkerConfigArgs):
            rms_norm_eps = args.text_config.rms_norm_eps
        else:
            raise NotImplementedError

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen2_5OmniAttention(
            args,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen2MLP(
            args, hidden_size=hidden_size, intermediate_size=intermediate_size
        )

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
    def __init__(self, args: TalkerConfigArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.embedding_size)
        self.layers = [
            Qwen2_5OmniDecoderLayer(
                args,
                layer_idx=i,
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                num_key_value_heads=args.num_key_value_heads,
                head_dim=args.head_dim,
                intermediate_size=args.intermediate_size,
            )
            for i in range(args.num_hidden_layers)
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
    def __init__(self, args: TalkerConfigArgs):
        super().__init__()

        self.model = Qwen2_5_TalkerModel(args)
        self.codec_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.thinker_to_talker_proj = nn.Linear(
            args.embedding_size, args.hidden_size, bias=True
        )

    def __call__(self, inputs, mask=None, cache=None, thinker_output=None):
        h, cache = self.model(inputs, mask, cache)
        return h, cache


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        # if channels % 2 != 0:
        #     raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        # log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        # inv_timescales = torch.exp(
        #     -log_timescale_increment * torch.arange(channels // 2)
        # ).float()
        # scaled_time = (
        #     torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        # )
        # self.register_buffer(
        #     "positional_embedding",
        #     torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
        #     persistent=False,
        # )


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        args: ThinkerConfigArgs,
    ):
        super().__init__()

        self.embed_dim = args.audio_config.d_model
        self.num_heads = args.audio_config.encoder_attention_heads
        self.dropout = args.audio_config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


class Qwen2_5OmniAudioEncoderLayer(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()

        self.embed_dim = args.audio_config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(args)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = args.audio_config.dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout = args.audio_config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, args.audio_config.encoder_ffn_dim)
        self.fc2 = nn.Linear(args.audio_config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)


class Conv1dReshaped(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # transpose the dimensions of the weights to match the tensor shape
        self.weight = self.weight.transpose(0, 2, 1)

    def __call__(self, x):
        # un-transpose the weights before call to match MLX expectation
        y = mx.conv1d(
            x,
            self.weight.transpose(0, 2, 1),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class Conv3dReshaped(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # transpose the dimensions of the weights to match the tensor shape
        self.weight = self.weight.transpose(0, 4, 1, 2, 3)

    def __call__(self, x):
        # un-transpose the weights before call to match MLX expectation
        y = mx.conv3d(
            x,
            self.weight.transpose(0, 2, 3, 4, 1),
            self.stride,
            self.padding,
            self.dilation,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class ConvTranspose1dReshaped(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # transpose the dimensions of the weights to match the tensor shape
        self.weight = self.weight.transpose(2, 0, 1)

    def __call__(self, x):
        # un-transpose the weights before call to match MLX expectation
        y = mx.conv_transpose1d(
            x,
            self.weight.transpose(1, 2, 0),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class Qwen2_5OmniAudioEncoder(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()
        self.dropout = args.audio_config.dropout

        embed_dim = args.audio_config.d_model
        self.num_mel_bins = args.audio_config.num_mel_bins
        self.max_source_positions = args.audio_config.max_source_positions
        self.embed_scale = (
            math.sqrt(embed_dim) if args.audio_config.scale_embedding else 1.0
        )
        self.n_window = args.audio_config.n_window
        self.conv1 = Conv1dReshaped(
            self.num_mel_bins, embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = Conv1dReshaped(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.audio_bos_eos_token = nn.Embedding(2, args.audio_config.output_dim)
        self.layers = [
            Qwen2_5OmniAudioEncoderLayer(args)
            for _ in range(args.audio_config.encoder_layers)
        ]
        self.ln_post = nn.LayerNorm(args.audio_config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(args.audio_config.d_model, args.audio_config.output_dim)
        self.gradient_checkpointing = False


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.variance_epsilon = eps


class Qwen2_5OmniVisionBlock(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()

        hidden_size = args.vision_config.hidden_size
        intermediate_size = args.vision_config.intermediate_size
        # Using thinker's rms_norm_eps for now
        rms_norm_eps = args.text_config.rms_norm_eps

        self.norm1 = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm2 = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = nn.Module()
        self.attn.q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn.k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn.v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)

    def __call__(self, x):
        # Simplified implementation
        return x


class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, args: ThinkerConfigArgs) -> None:
        super().__init__()
        dim = args.vision_config.out_hidden_size
        context_dim = args.vision_config.hidden_size
        spatial_merge_size = args.vision_config.spatial_merge_size
        # Assuming Thinker's text_config rms_norm_eps is used here too
        rms_norm_eps = args.text_config.rms_norm_eps

        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.RMSNorm(context_dim, eps=rms_norm_eps)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x):  # Added x parameter
        pass


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()
        self.patch_size = args.vision_config.patch_size
        self.temporal_patch_size = args.vision_config.temporal_patch_size
        self.in_channels = args.vision_config.in_channels
        self.embed_dim = (
            args.vision_config.hidden_size
        )  # embed_dim is vision hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = Conv3dReshaped(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )


class Qwen2_5OmniVisionEncoder(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()

        self.blocks = [
            Qwen2_5OmniVisionBlock(args) for _ in range(args.vision_config.depth)
        ]
        self.patch_embed = Qwen2_5_VisionPatchEmbed(args)

        self.merger = Qwen2_5OmniPatchMerger(args)

    def __call__(self, x):
        # Simplified implementation
        return x


class Qwen2_5_ThinkerModel(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()

        hidden_size = args.text_config.hidden_size
        intermediate_size = args.text_config.intermediate_size
        head_dim = hidden_size // args.text_config.num_attention_heads

        self.embed_tokens = nn.Embedding(args.text_config.vocab_size, hidden_size)
        self.layers = [
            Qwen2_5OmniDecoderLayer(
                args,
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=args.text_config.num_attention_heads,
                num_key_value_heads=args.text_config.num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
            )
            for layer_idx in range(args.text_config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=args.text_config.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None):
        return inputs


class Qwen2_5_Thinker(nn.Module):
    def __init__(self, args: ThinkerConfigArgs):
        super().__init__()
        self.model = Qwen2_5_ThinkerModel(args)
        self.audio_tower = Qwen2_5OmniAudioEncoder(args)
        self.visual = Qwen2_5OmniVisionEncoder(args)
        self.lm_head = nn.Linear(
            args.text_config.hidden_size, args.text_config.vocab_size, bias=False
        )

    def __call__(
        self, inputs, mask=None, cache=None, audio_input=None, visual_input=None
    ):
        # Simplified implementation
        h, cache = self.model(inputs, mask, cache)
        return h, cache


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = mx.zeros(in_features) * alpha
        self.beta = mx.zeros(in_features) * alpha

        self.no_div_by_zero = 0.000000001

    def __call__(self, *args, **kwargs):
        pass


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Generates a 1D Kaiser-windowed sinc filter.

    Args:
        cutoff (float): Normalized cutoff frequency (0 to 0.5).
        half_width (float): Transition bandwidth.
        kernel_size (int): Number of filter taps.

    Returns:
        torch.Tensor: A tensor of shape (1, 1, kernel_size) representing the filter.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute Kaiser window parameters
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch_kaiser_window(
        kernel_size, beta=beta, periodic=False, dtype=torch_kaiser_window_dtype
    )
    kaiser_window = mx.array(kaiser_window.float().numpy())

    # Compute time indices
    if is_even:
        time_indices = torch_arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch_arange(kernel_size) - half_size

    # Compute sinc filter
    if cutoff == 0:
        return mx.zeros((1, 1, kernel_size), dtype=mx.float32)  # Ensures correct shape

    sinc_filter = torch_sinc(2 * cutoff * time_indices)
    sinc_filter = mx.array(sinc_filter.float().numpy())

    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    # Normalize to ensure sum = 1 (avoid leakage of constant component)
    normalized_filter /= normalized_filter.sum()

    # return normalized_filter.view(1, 1, kernel_size)
    return mx.expand_dims(mx.expand_dims(normalized_filter.flatten(), 0), 0)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )

        self._filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        # self.register_buffer("filter", filter, persistent=False)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        self._filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise ValueError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)


class AMPBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__()

        self.convs1 = [
            Conv1dReshaped(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[0],
                padding=self._get_padding(kernel_size, dilation[0]),
            ),
            Conv1dReshaped(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[1],
                padding=self._get_padding(kernel_size, dilation[1]),
            ),
            Conv1dReshaped(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[2],
                padding=self._get_padding(kernel_size, dilation[2]),
            ),
        ]

        self.convs2 = [
            Conv1dReshaped(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=self._get_padding(kernel_size, 1),
            ),
            Conv1dReshaped(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=self._get_padding(kernel_size, 1),
            ),
            Conv1dReshaped(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=self._get_padding(kernel_size, 1),
            ),
        ]

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        self.activations = [
            TorchActivation1d(activation=SnakeBeta(channels))
            for _ in range(self.num_layers)
        ]

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)


class Qwen2_5OmniToken2WavBigVGANModel(nn.Module):
    def __init__(self, args: Token2WavConfigArgs):
        super().__init__()

        cfg = args.bigvgan_config
        self.num_residual_blocks = len(cfg.resblock_kernel_sizes)
        self.num_upsample_layers = len(cfg.upsample_rates)

        self.conv_pre = Conv1dReshaped(
            cfg.mel_dim, cfg.upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = [
            [
                ConvTranspose1dReshaped(
                    cfg.upsample_initial_channel // (2**layer_idx),
                    cfg.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            ]
            for layer_idx, (stride, kernel_size) in enumerate(
                zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)
            )
        ]

        self.resblocks = [
            AMPBlock(
                cfg.upsample_initial_channel // (2 ** (layer_idx + 1)),
                kernel_size,
                dilation,
            )
            for layer_idx in range(self.num_upsample_layers)
            for kernel_size, dilation in zip(
                cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes
            )
        ]

        post_channel_size = cfg.upsample_initial_channel // (
            2**self.num_upsample_layers
        )
        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(post_channel_size)
        )
        self.conv_post = Conv1dReshaped(
            post_channel_size,
            1,
            7,
            1,
            padding=3,
            bias=False,
        )


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
    def __init__(self, args: Token2WavConfigArgs):
        super().__init__()

        cfg = args.dit_config
        self.dim = cfg.dim
        self.heads = cfg.heads
        self.inner_dim = cfg.head_dim * cfg.heads
        self.dropout = cfg.dropout
        # self._attn_implementation = config._attn_implementation # If needed
        self.is_causal = False

        self.to_q = nn.Linear(self.dim, self.inner_dim)
        self.to_k = nn.Linear(self.dim, self.inner_dim)
        self.to_v = nn.Linear(self.dim, self.inner_dim)

        self.to_out = [nn.Linear(self.inner_dim, self.dim), nn.Dropout(self.dropout)]


class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6, bias=True)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)


class DiTDecoderLayer(nn.Module):
    def __init__(
        self, args: Token2WavConfigArgs, look_ahead_block=0, look_backward_block=0
    ):
        super().__init__()

        cfg = args.dit_config
        hidden_size = cfg.dim

        self.attn_norm = Qwen2_5_OmniAdaLayerNormZero(hidden_size)

        self.attn = DiTAttention(args)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.ff = DiTMLP(dim=hidden_size, mult=cfg.ff_mult, dropout=cfg.dropout)


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


class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = Conv1dReshaped(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            # padding="same",  # NM: fix this?
            # padding_mode="reflect",
        )
        self.activation = nn.ReLU()


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = [
            TimeDelayNetBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for i in range(scale - 1)
        ]
        self.scale = scale


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1dReshaped(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            # padding="same",  # TODO: fix this
            # padding_mode="reflect",
        )


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = Conv1dReshaped(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            # padding="same",  # NM: fix this?
            # padding_mode="reflect",
        )
        self.relu = nn.ReLU()  # TODO: this has to be in-place relu
        self.conv2 = Conv1dReshaped(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            # padding="same",  # NM: fix this?
            # padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()


class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SqueezeExcitationBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)


class ECAPA_TimeDelayNet(nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    """

    def __init__(self, args: Token2WavConfigArgs):
        super().__init__()

        cfg = args.dit_config
        enc_channels = cfg.enc_channels
        enc_kernel_sizes = cfg.enc_kernel_sizes
        enc_dilations = cfg.enc_dilations
        mel_dim = cfg.mel_dim
        enc_res2net_scale = cfg.enc_res2net_scale
        enc_se_channels = cfg.enc_se_channels
        enc_attention_channels = cfg.enc_attention_channels
        enc_dim = cfg.enc_dim

        if len(enc_channels) != len(enc_kernel_sizes) or len(enc_channels) != len(
            enc_dilations
        ):
            raise ValueError(
                "enc_channels, enc_kernel_sizes and enc_dilations should have same length"
            )
        self.channels = enc_channels
        self.blocks = []

        # The initial TDNN layer
        self.blocks.append(
            TimeDelayNetBlock(
                mel_dim,
                enc_channels[0],
                enc_kernel_sizes[0],
                enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    enc_channels[i - 1],
                    enc_channels[i],
                    res2net_scale=enc_res2net_scale,
                    se_channels=enc_se_channels,
                    kernel_size=enc_kernel_sizes[i],
                    dilation=enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TimeDelayNetBlock(
            enc_channels[-1],
            enc_channels[-1],
            enc_kernel_sizes[-1],
            enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            enc_channels[-1],
            attention_channels=enc_attention_channels,
        )

        # Final linear transformation
        self.fc = Conv1dReshaped(
            in_channels=enc_channels[-1] * 2,
            out_channels=enc_dim,
            kernel_size=1,
            # padding="same",  # TODO: fix this
            # padding_mode="reflect",
        )


class DiTInputEmbedding(nn.Module):
    def __init__(self, args: Token2WavConfigArgs):
        super().__init__()

        cfg = args.dit_config
        self.proj = nn.Linear(
            cfg.mel_dim + cfg.enc_dim + cfg.enc_emb_dim + cfg.emb_dim,
            cfg.dim,
        )
        self.spk_encoder = ECAPA_TimeDelayNet(args)


# Transformer backbone using DiT blocks
class DiTCodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)


class Qwen2_5_OmniAdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)


class Code2WavDITModel(nn.Module):
    def __init__(self, args: Token2WavConfigArgs):  # Expect Token2WavConfigArgs
        super().__init__()

        # Use parameters from args.dit_config
        cfg = args.dit_config
        hidden_size = cfg.dim

        self.input_embed = DiTInputEmbedding(args)

        self.text_embed = DiTCodecEmbedding(cfg.num_embeds, cfg.emb_dim, cfg.repeats)

        self.time_embed = DiTTimestepEmbedding(hidden_size)

        self.rotary_embed = nn.Module()  # Placeholder

        self.transformer_blocks = [
            DiTDecoderLayer(
                args,
                look_ahead_block=1 if i in cfg.look_ahead_layers else 0,
                look_backward_block=1 if i in cfg.look_backward_layers else 0,
            )
            for i in range(cfg.depth)
        ]

        self.norm_out = Qwen2_5_OmniAdaLayerNormZero_Final(hidden_size)
        self.proj_out = nn.Linear(hidden_size, cfg.mel_dim)

    def __call__(self, x):
        # Simplified implementation
        return x


class Qwen2_5_Token2wav(nn.Module):
    def __init__(self, args: Token2WavConfigArgs):
        super().__init__()
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel(args)
        self.code2wav_dit_model = Code2WavDITModel(args)

    def __call__(self, x):
        # Simplified implementation
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.talker = Qwen2_5_Talker(args.talker)
        self.thinker = Qwen2_5_Thinker(args.thinker)
        self.token2wav = Qwen2_5_Token2wav(args.token2wav)

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
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "attention.rope.inv_freq" not in k
        }
        weights = {k: v for k, v in weights.items() if "rotary_embed.inv_freq" not in k}

        # Remove unused precomputed sampling filters
        weights = {k: v for k, v in weights.items() if "downsample.filter" not in k}
        weights = {k: v for k, v in weights.items() if "upsample.filter" not in k}

        return weights

    @property
    def layers(self):
        return self.talker.model.layers
