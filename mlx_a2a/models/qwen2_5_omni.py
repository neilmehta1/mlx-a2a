from dataclasses import dataclass
import mlx.nn as nn
import mlx.core as mx

from mlx_lm.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str


class Qwen2_5_TalkerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()


class Qwen2_5_Talker(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Qwen2_5_TalkerModel
        self.embed_tokens = nn.Embedding(1, 1)  # NM: fix


class Qwen2_5_Thinker(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()


class Qwen2_5_Token2wav(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()


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
        pass

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers
