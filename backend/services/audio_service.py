import os
import sys
import logging
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = os.path.join(ROOT, "scratch", "voxceleb_trainer")
MODEL_PATH = os.path.join(ROOT, "scratch", "baseline_lite_ap.model")

# Append voxceleb_trainer to sys.path so we can load ResNetSE34L
if TRAINER_PATH not in sys.path:
    sys.path.append(TRAINER_PATH)

# Global model placeholders for lazy-loading
_resnet_model = None
_whisper_model = None

# Enrolled speaker profiles (embeddings cache)
# maps speaker_name -> embedding tensor
_enrolled_speakers: Dict[str, torch.Tensor] = {}


def load_speaker_model():
    """Lazy load the ResNetSE34L speaker verification model."""
    global _resnet_model
    if _resnet_model is not None:
        return _resnet_model

    try:
        from models.ResNetSE34L import MainModel
        log.info("Initializing ResNetSE34L speaker verification model...")
        model = MainModel(nOut=512, encoder_type='SAP')
        model.eval()

        if not os.path.exists(MODEL_PATH):
            log.warning("ResNet model weights not found at %s. Verification will fail.", MODEL_PATH)
            return None

        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace("__S__.", "").replace("module.", "")
            state_dict[new_key] = v

        model.load_state_dict(state_dict)
        _resnet_model = model
        log.info("ResNetSE34L model loaded successfully.")
        return _resnet_model
    except Exception as e:
        log.error("Failed to load speaker verification model: %s", e)
        return None


def load_whisper_model():
    """Lazy load the faster-whisper transcription model."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
        model_size = os.environ.get("WHISPER_MODEL_SIZE", "tiny.en")
        log.info("Loading faster-whisper model: %s ...", model_size)
        # Force CPU and float32 for clean execution inside standard Docker containers
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="float32")
        log.info("faster-whisper model %s loaded successfully.", model_size)
        return _whisper_model
    except Exception as e:
        log.error("Failed to load faster-whisper model: %s", e)
        return None


def get_audio_embedding(audio_path: str) -> Optional[torch.Tensor]:
    """Load audio file, resample to 16kHz mono, and extract ResNet embedding."""
    model = load_speaker_model()
    if not model:
        return None

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16000Hz mono
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ResNet expects 2D tensor (1, num_samples)
        waveform = waveform.squeeze(0).unsqueeze(0)

        with torch.no_grad():
            embedding = model(waveform)
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.squeeze(0)
    except Exception as e:
        log.error("Failed to extract speaker embedding for %s: %s", audio_path, e)
        return None


def enroll_speaker(speaker_name: str, audio_path: str) -> bool:
    """Enroll a new speaker profile using a reference audio file."""
    emb = get_audio_embedding(audio_path)
    if emb is not None:
        _enrolled_speakers[speaker_name] = emb
        log.info("Successfully enrolled speaker: %s", speaker_name)
        return True
    return False


def identify_speaker(audio_path: str, threshold: float = 0.40) -> str:
    """Compare audio embedding against enrolled profiles and return the closest match name."""
    emb = get_audio_embedding(audio_path)
    if emb is None or not _enrolled_speakers:
        return "unknown"

    best_name = "unknown"
    best_score = -1.0

    for name, ref_emb in _enrolled_speakers.items():
        # Cosine similarity between 1D normalized tensors is the dot product
        score = torch.dot(emb, ref_emb).item()
        if score > best_score:
            best_score = score
            best_name = name

    log.info("Speaker verification best match: %s (score: %.4f)", best_name, best_score)
    if best_score >= threshold:
        return best_name
    return "unknown"


def transcribe_audio(audio_path: str) -> Tuple[str, float]:
    """Transcribe audio file using faster-whisper and return text + average segment probability."""
    model = load_whisper_model()
    if not model:
        return "", 0.0

    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        text_segments = []
        probs = []
        for segment in segments:
            text_segments.append(segment.text)
            probs.append(segment.avg_logprob)

        full_text = " ".join(text_segments).strip()
        # Convert log prob back to standard probability
        import math
        avg_prob = math.exp(sum(probs) / len(probs)) if probs else 1.0
        return full_text, avg_prob
    except Exception as e:
        log.error("Failed to transcribe audio file %s: %s", audio_path, e)
        return "", 0.0
