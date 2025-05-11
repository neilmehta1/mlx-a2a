import unittest
import numpy as np
import mlx.core as mx
import json
from pathlib import Path

from mlx_a2a.convert import load_model


class TestQwen25Omni(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path("Qwen/Qwen2.5-Omni-3B-MLX")
        cls.model, _ = load_model(model_path)
        cls.config = cls.model.thinker.args

    def test_get_rope_index(self):
        """Test the get_rope_index function using captured IO data."""
        # Load the captured inputs and outputs
        inputs = np.load("io_capture_get_rope_index/get_rope_index_inputs.npz")
        outputs = np.load("io_capture_get_rope_index/get_rope_index_outputs.npz")

        # Convert inputs to MLX arrays
        input_ids = mx.array(inputs["input_ids"])
        image_grid_thw = mx.array(inputs["image_grid_thw"])
        attention_mask = mx.array(inputs["attention_mask"])
        audio_seqlens = mx.array(inputs["audio_seqlens"])

        # Load metadata for None and boolean inputs
        with open("io_capture_get_rope_index/get_rope_index_metadata.json", "r") as f:
            metadata = json.load(f)

        # Call the get_rope_index function
        position_ids, mrope_position_deltas = self.model.get_rope_index(
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

    def test_get_rope_index_video_with_audio(self):
        """Test the get_rope_index function with video and audio inputs."""
        # Test spatial_merge_size = 1 to match hf-transformers test
        orig_spatial_merge_size = (
            self.model.thinker.args.vision_config.spatial_merge_size
        )
        self.model.thinker.args.vision_config.spatial_merge_size = 1

        try:
            # Original PyTorch: image_grid_thw = torch.empty((0, 3), dtype=torch.long)
            image_grid_thw = mx.random.randint(
                -1000000000, 1000000000, (0, 3), dtype=mx.int64
            )

            # Original PyTorch: video_grid_thw = torch.tensor([[3, 2, 2]], dtype=torch.long)
            # 3 * 2 * 2 = 12 video tokens per frame in video_grid_thw
            video_grid_thw = mx.array([[3, 2, 2]], dtype=mx.int64)

            # Original PyTorch: audio_seqlens = torch.tensor([300], dtype=torch.long)
            # num_audio_tokens = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
            # i.e.: 300 audio_seqlen -> 75 audio tokens
            audio_seqlens = mx.array([300], dtype=mx.int64)

            # Original PyTorch: second_per_grids = torch.tensor([1.0], dtype=torch.float)
            second_per_grids = mx.array([1.0], dtype=mx.float32)

            use_audio_in_video = True

            # Original PyTorch: expected_position_ids = torch.tensor(...)
            # fmt: off
            expected_position_ids_data = [
                [[
                    0,  1, # text
                    2,  2, # vision_bos + audio_bos

                    # video chunk
                    3,  3,  3,  3,
                    28, 28, 28, 28,

                    # audio chunk
                    3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    45, 46, 47, 48, 49, 50, 51, 52,

                    # video chunk
                    53, 53, 53, 53,

                    # audio chunk
                    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,

                    78, 78, # audio_eos + vision_eos
                    79, 80, # text
                ]],
                [[
                    0,  1, # text
                    2,  2, # vision_bos + audio_bos

                    # video chunk
                    3,  3,  4,  4,
                    3,  3,  4,  4,

                    # audio chunk
                    3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    45, 46, 47, 48, 49, 50, 51, 52,

                    # video chunk
                    3,  3,  4,  4,

                    # audio chunk
                    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,

                    78, 78, # audio_eos + vision_eos
                    79, 80, # text
                ]],
                [[
                    0,  1, # text
                    2,  2, # vision_bos + audio_bos

                    # video chunk
                    3,  4,  3,  4,
                    3,  4,  3,  4,

                    # audio chunk
                    3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    45, 46, 47, 48, 49, 50, 51, 52,

                    # video chunk
                    3,  4,  3,  4,

                    # audio chunk
                    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,

                    78, 78, # audio_eos + vision_eos
                    79, 80, # text
                ]],
            ]
            # fmt: on
            expected_position_ids = mx.array(expected_position_ids_data)

            input_ids_list = (
                [
                    100,
                    101,
                ]
                + [
                    self.config.vision_start_token_id,
                    self.config.audio_start_token_id,
                ]
                # 1st chunk: 8 video tokens (2*2*2), 50 audio tokens
                + [self.config.video_token_index] * (2 * 2 * 2)
                + [self.config.audio_token_index] * 50
                +
                # 2nd chunk: 4 video tokens (1*2*2), 25 audio tokens
                [self.config.video_token_index] * (1 * 2 * 2)
                + [self.config.audio_token_index] * 25
                + [
                    self.config.audio_end_token_id,
                    self.config.vision_end_token_id,
                ]
                + [
                    102,
                    103,
                ]
            )
            input_ids = mx.array([input_ids_list], dtype=mx.int64)

            position_ids, mrope_position_deltas = self.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=None,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_seqlens,
                second_per_grids=second_per_grids,
            )

            self.assertTrue(mx.allclose(position_ids, expected_position_ids))
            # mrope_position_deltas is returned but not asserted in the original PyTorch test for this case.
        finally:
            self.model.thinker.args.vision_config.spatial_merge_size = (
                orig_spatial_merge_size
            )


if __name__ == "__main__":
    unittest.main()
