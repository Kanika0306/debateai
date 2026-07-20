import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from backend.services import audio_service


@pytest.fixture
def mock_torchaudio():
    with patch("torchaudio.load") as mock_load, \
         patch("torchaudio.transforms.Resample") as mock_resample:
        
        # Mock load returning a dummy 16kHz mono waveform
        dummy_waveform = torch.zeros((1, 16000), dtype=torch.float32)
        mock_load.return_value = (dummy_waveform, 16000)
        
        mock_resampler_instance = MagicMock()
        mock_resampler_instance.return_value = dummy_waveform
        mock_resample.return_value = mock_resampler_instance
        
        yield mock_load, mock_resample


@pytest.fixture
def mock_speaker_model():
    with patch("backend.services.audio_service.load_speaker_model") as mock_load_model:
        mock_model = MagicMock()
        # Mock forward pass returning 512-dim embedding
        dummy_embedding = torch.randn((1, 512), dtype=torch.float32)
        mock_model.return_value = dummy_embedding
        mock_load_model.return_value = mock_model
        yield mock_load_model, mock_model


@pytest.fixture
def mock_whisper_model():
    with patch("backend.services.audio_service.load_whisper_model") as mock_load_model:
        mock_model = MagicMock()
        # Mock transcribe returning list of dummy segments
        dummy_segment = MagicMock()
        dummy_segment.text = "Hello world from audio transcription test."
        dummy_segment.avg_logprob = -0.1
        
        mock_model.transcribe.return_value = ([dummy_segment], None)
        mock_load_model.return_value = mock_model
        yield mock_load_model, mock_model


def test_audio_embedding_generation(mock_torchaudio, mock_speaker_model):
    """Test that embedding generation resamples, processes, and returns a 512-dim tensor."""
    emb = audio_service.get_audio_embedding("dummy_path.wav")
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (512,)
    # Verify values are L2-normalized
    norm = torch.norm(emb, p=2).item()
    assert pytest.approx(norm) == 1.0


def test_speaker_enrollment_and_identification(mock_torchaudio, mock_speaker_model):
    """Test enrolling a speaker profile and matching a verified embedding."""
    # Reset enrolled speakers
    audio_service._enrolled_speakers = {}

    # Mock get_audio_embedding to return a fixed pattern for Speaker X
    fixed_emb_x = torch.zeros((512,), dtype=torch.float32)
    fixed_emb_x[0] = 1.0  # L2 normalized

    fixed_emb_y = torch.zeros((512,), dtype=torch.float32)
    fixed_emb_y[1] = 1.0  # L2 normalized

    with patch("backend.services.audio_service.get_audio_embedding") as mock_get_emb:
        # First call: enroll Speaker_X
        # Second call: enroll Speaker_Y
        # Third call: verify Speaker_X matches closest
        mock_get_emb.side_effect = [fixed_emb_x, fixed_emb_y, fixed_emb_x]

        assert audio_service.enroll_speaker("Speaker_X", "x.wav")
        assert audio_service.enroll_speaker("Speaker_Y", "y.wav")

        matched_speaker = audio_service.identify_speaker("test.wav", threshold=0.40)
        assert matched_speaker == "Speaker_X"


def test_speaker_identification_below_threshold(mock_torchaudio, mock_speaker_model):
    """Test speaker identification falls back to 'unknown' if similarity is below threshold."""
    audio_service._enrolled_speakers = {}

    fixed_emb_x = torch.zeros((512,), dtype=torch.float32)
    fixed_emb_x[0] = 1.0

    # Orthogonal embedding (dot product = 0.0)
    fixed_emb_unknown = torch.zeros((512,), dtype=torch.float32)
    fixed_emb_unknown[1] = 1.0

    with patch("backend.services.audio_service.get_audio_embedding") as mock_get_emb:
        mock_get_emb.side_effect = [fixed_emb_x, fixed_emb_unknown]

        assert audio_service.enroll_speaker("Speaker_X", "x.wav")
        matched_speaker = audio_service.identify_speaker("unknown.wav", threshold=0.40)
        assert matched_speaker == "unknown"


def test_transcription(mock_whisper_model):
    """Test transcription using the whisper helper returns combined text and probability."""
    text, prob = audio_service.transcribe_audio("dummy_path.wav")
    assert text == "Hello world from audio transcription test."
    # Log prob of -0.1 translates to e^-0.1 ~ 0.9048 probability
    assert pytest.approx(prob, rel=1e-3) == 0.9048
