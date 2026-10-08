"""ASRModel.generate_streaming: token-by-token output matching generate()."""

import torch

from tiny_audio.asr_modeling import ASRModel


class TestGenerateStreaming:
    """generate_streaming yields partial transcript pieces."""

    def test_streaming_yields_strings(self, base_asr_model: ASRModel) -> None:
        input_features = torch.zeros(1, 80, 3000)
        audio_attention_mask = torch.ones(1, 3000, dtype=torch.long)

        outputs = list(
            base_asr_model.generate_streaming(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                max_new_tokens=4,
            )
        )
        # Each yielded piece is a string (possibly empty)
        for piece in outputs:
            assert isinstance(piece, str)

    def test_streaming_matches_generate(self, base_asr_model: ASRModel) -> None:
        """Streaming decodes the same tokens as generate(), without the prompt."""
        torch.manual_seed(0)
        input_features = torch.randn(1, 80, 3000)
        audio_attention_mask = torch.ones(1, 3000, dtype=torch.long)

        ids = base_asr_model.generate(
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
            max_new_tokens=6,
        )
        assert isinstance(ids, torch.Tensor)
        expected = base_asr_model.tokenizer.decode(ids[0], skip_special_tokens=True)
        streamed = "".join(
            base_asr_model.generate_streaming(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                max_new_tokens=6,
            )
        )
        assert expected
        assert streamed == expected
