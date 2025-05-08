import unittest
import numpy as np
import mlx.core as mx
import json
from pathlib import Path

from mlx_a2a.convert import load_model


class TestQwen25Omni(unittest.TestCase):
    def test_get_rope_index(self):
        # Load the captured inputs and outputs
        inputs = np.load("io_capture_qwen2_5_omni_rope_index/get_rope_index_inputs.npz")
        outputs = np.load(
            "io_capture_qwen2_5_omni_rope_index/get_rope_index_outputs.npz"
        )

        # Convert inputs to MLX arrays
        input_ids = mx.array(inputs["input_ids"])
        image_grid_thw = mx.array(inputs["image_grid_thw"])
        attention_mask = mx.array(inputs["attention_mask"])
        audio_seqlens = mx.array(inputs["audio_seqlens"])

        # Load metadata for None and boolean inputs
        with open(
            "io_capture_qwen2_5_omni_rope_index/get_rope_index_metadata.json", "r"
        ) as f:
            metadata = json.load(f)

        # Create a model instance by loading from Qwen
        model_path = Path("Qwen/Qwen2.5-Omni-3B-MLX")
        model, _ = load_model(model_path)

        # Call the get_rope_index function
        position_ids, mrope_position_deltas = model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            use_audio_in_video=metadata["bool_inputs"]["use_audio_in_video"],
            audio_seqlens=audio_seqlens,
            video_grid_thw=None,
            second_per_grids=None,
        )

        # Convert the expected outputs to MLX arrays
        expected_position_ids = mx.array(outputs["output_0"])
        expected_mrope_position_deltas = mx.array(outputs["output_1"])

        # Check that the outputs match the expected values
        self.assertTrue(mx.array_equal(position_ids, expected_position_ids))
        self.assertTrue(
            mx.array_equal(mrope_position_deltas, expected_mrope_position_deltas)
        )


if __name__ == "__main__":
    unittest.main()
