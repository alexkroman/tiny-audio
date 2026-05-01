"""Audio-token counting + Qwen3 chat-template parity tests for the MLX processor."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def _whisper_extractor():
    """Helper: load the GLM-ASR feature extractor (matches the PT inference path)."""
    from transformers import AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
    fe.padding = False
    return fe


def _qwen3_tokenizer():
    """Helper: load Qwen3-0.6B tokenizer with the <audio> token added."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok.add_special_tokens({"additional_special_tokens": ["<audio>"]})
    return tok


@pytest.mark.parametrize("audio_seconds", [1.0, 3.5, 7.0, 15.0])
def test_audio_token_count_matches_pt_pipeline(audio_seconds):
    """compute_num_audio_tokens must match the count the PT ASRProcessor would generate."""
    from tests.conftest import MockProjectorConfig
    from tiny_audio.asr_processing import ASRProcessor
    from tiny_audio.mlx.processor import compute_num_audio_tokens
    from tiny_audio.mlx.projector import MLXMLPProjector
    from tiny_audio.projectors import MLPAudioProjector

    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(16000 * audio_seconds)).astype(np.float32)

    fe = _whisper_extractor()
    tok = _qwen3_tokenizer()

    # GLM-ASR encoder dim = 1280, Qwen3-0.6B llm_dim = 1024,
    # embedded.yaml uses projector_hidden_dim=1024, projector_pool_stride=4
    cfg = MockProjectorConfig(
        encoder_dim=1280,
        llm_dim=1024,
        projector_pool_stride=4,
        projector_hidden_dim=1024,
    )
    pt_proj = MLPAudioProjector(cfg)
    pt_processor = ASRProcessor(feature_extractor=fe, tokenizer=tok, projector=pt_proj)
    pt_inputs = pt_processor(audio=audio)
    pt_audio_token_count = int(
        (pt_inputs["input_ids"] == tok.convert_tokens_to_ids("<audio>")).sum().item()
    )

    mlx_proj = MLXMLPProjector(encoder_dim=1280, llm_dim=1024, hidden_dim=1024, pool_stride=4)
    mlx_count = compute_num_audio_tokens(
        audio_len=len(audio),
        encoder_conv_layers=[(1, 3, 1), (1, 3, 2)],
        projector=mlx_proj,
    )

    assert mlx_count == pt_audio_token_count, (
        f"audio_seconds={audio_seconds}, mlx={mlx_count}, pt={pt_audio_token_count}"
    )


def test_chat_template_input_ids_match_pt():
    """Chat-template tokenization must be byte-identical to the PT path."""
    from tiny_audio.mlx.processor import build_prompt_input_ids

    tok = _qwen3_tokenizer()

    from tiny_audio.mlx.processor import TRANSCRIBE_PROMPT

    num_audio = 17
    audio_placeholder = "<audio>" * num_audio
    user_content = audio_placeholder
    if TRANSCRIBE_PROMPT:
        user_content += " " + TRANSCRIBE_PROMPT
    messages = [{"role": "user", "content": user_content}]

    pt_out = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np",
        enable_thinking=False,
    )
    if isinstance(pt_out, np.ndarray):
        pt_ids = pt_out
    elif hasattr(pt_out, "input_ids"):
        pt_ids = np.asarray(pt_out["input_ids"])
    else:
        pt_ids = np.asarray(pt_out)
    if pt_ids.ndim == 1:
        pt_ids = pt_ids[None, :]

    mlx_ids = build_prompt_input_ids(tok, num_audio_tokens=num_audio, system_prompt="")
    np.testing.assert_array_equal(np.array(mlx_ids), pt_ids)


def test_chat_template_with_system_prompt():
    """System prompt is added when non-empty and prepended to the user message."""
    from tiny_audio.mlx.processor import build_prompt_input_ids

    tok = _qwen3_tokenizer()

    ids_no_sys = build_prompt_input_ids(tok, num_audio_tokens=5, system_prompt="")
    ids_with_sys = build_prompt_input_ids(
        tok, num_audio_tokens=5, system_prompt="You are an ASR engine."
    )
    # With system prompt, prompt should be longer
    assert ids_with_sys.shape[1] > ids_no_sys.shape[1]


def test_compute_encoder_output_length_matches_glm_conv():
    """Verify the conv formula matches GLM-ASR's actual subsampling: stride 1 then stride 2."""
    from tiny_audio.mlx.processor import compute_encoder_output_length

    glm_layers = [(1, 3, 1), (1, 3, 2)]
    # mel_len = 100 (1 second of audio). After conv1 (s=1): 100. After conv2 (s=2): (100+1)//2 = 50.
    assert compute_encoder_output_length(100, glm_layers) == 50
    # mel_len = 1000 (10 seconds). conv1 -> 1000, conv2 -> 500.
    assert compute_encoder_output_length(1000, glm_layers) == 500
    # mel_len = 7 (boundary). conv1 -> 7, conv2 -> (7+1)//2 = 4.
    assert compute_encoder_output_length(7, glm_layers) == 4
