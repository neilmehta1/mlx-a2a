# Copyright © 2023-2024 Apple Inc.

import argparse
import glob
import json
from pathlib import Path
import shutil
import sys
import importlib
import pkgutil
import logging
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.utils import (
    save_weights,
    get_model_path,
    load_model,
    load_tokenizer,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def save(
    dst_path: Union[str, Path],
    src_path: Union[str, Path],
    weights: Dict[str, mx.array],
    tokenizer: TokenizerWrapper,
    config: Dict[str, Any],
    hf_repo: Optional[str] = None,
    donate_weights: bool = True,
):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    save_weights(dst_path, weights, donate_weights=True)
    save_config(config, config_path=dst_path / "config.json")
    tokenizer.save_pretrained(dst_path)

    for p in ["*.py", "*.json"]:
        for file in glob.glob(str(src_path / p)):
            shutil.copy(file, dst_path)


MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model, config = load_model(model_path, lazy)
    tokenizer = load_tokenizer(
        model_path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, config, tokenizer


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    revision: Optional[str] = None,
):
    # Dynamically register models from mlx_a2a.models into mlx_lm.models namespace
    custom_models_path = Path(__file__).parent / "models"
    for module_info in pkgutil.iter_modules([str(custom_models_path)]):
        if not module_info.ispkg:  # Avoid registering packages like __pycache__
            custom_module_name = f"mlx_a2a.models.{module_info.name}"
            target_module_name = f"mlx_lm.models.{module_info.name}"
            try:
                module = importlib.import_module(custom_module_name)
                sys.modules[target_module_name] = module
                logging.info(f"Registered {custom_module_name} as {target_module_name}")
            except ImportError as e:
                logging.warning(
                    f"Could not import custom model {custom_module_name}: {e}"
                )

    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    dtype = config.get("torch_dtype", None)
    weights = dict(tree_flatten(model.parameters()))
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    del model
    save(
        mlx_path,
        model_path,
        weights,
        tokenizer,
        config,
        hf_repo=hf_path,
    )


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


# if __name__ == "__main__":
#     print(
#         "Calling `python -m mlx_lm.convert ...` directly is deprecated."
#         " Use `mlx_lm.convert ...` or `python -m mlx_lm convert ...` instead."
#     )
#     main()
