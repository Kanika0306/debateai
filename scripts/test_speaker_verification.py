"""
test_speaker_verification.py - Run speaker verification on VoxCeleb sample files
using the pre-trained Clova AI ResNetSE34L model.
"""
import os
import sys
import urllib.request
import torch
import torchaudio
import torch.nn.functional as F

# Append the voxceleb_trainer path to sys.path so we can import its models
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER_PATH = os.path.join(BASE, "scratch", "voxceleb_trainer")
sys.path.append(TRAINER_PATH)

from models.ResNetSE34L import MainModel

MODEL_URL = "http://www.robots.ox.ac.uk/~joon/data/baseline_lite_ap.model"
MODEL_PATH = os.path.join(BASE, "scratch", "baseline_lite_ap.model")
VOXCELEB_DIR = os.path.join(BASE, "data", "raw", "diarization", "voxceleb_sample")

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Pre-trained model already exists at {MODEL_PATH}")
        return
    print(f"Downloading pre-trained model weights from {MODEL_URL}...")
    print("This might take a minute...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

def load_audio(filename):
    path = os.path.join(VOXCELEB_DIR, filename)
    waveform, sample_rate = torchaudio.load(path)
    
    # Model expects 16000Hz mono audio
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Average channels if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform.squeeze(0)  # Return 1D tensor

def get_embedding(model, waveform):
    # Model expects input of shape (1, num_samples)
    waveform = waveform.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        embedding = model(waveform)
        
    # L2 Normalize the embedding
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0)

def main():
    print("=" * 60)
    print("  VoxCeleb Speaker Verification Demo")
    print("=" * 60)

    # 1. Download model weights
    download_model()

    # 2. Initialize ResNetSE34L model
    print("\nInitializing ResNetSE34L model...")
    model = MainModel(nOut=512, encoder_type='SAP')
    model.eval()

    # 3. Load model weights
    print("Loading model weights...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Map the model keys (legacy baseline baseline_lite_ap.model saves keys with __S__. prefix)
    state_dict = {}
    for k, v in checkpoint.items():
        # Remove __S__. prefix if present
        new_key = k.replace("__S__.", "").replace("module.", "")
        state_dict[new_key] = v
        
    model.load_state_dict(state_dict)
    print("Weights loaded successfully.")

    # 4. Load audio samples
    samples = [
        {"file": "sample_001_ident.wav", "name": "Sample 1 (Speaker A - ID: Y8hIVOBuels)"},
        {"file": "sample_002_verif.wav", "name": "Sample 2 (Speaker B - ID: 8jEAjG6SegY)"},
        {"file": "sample_003_verif.wav", "name": "Sample 3 (Speaker B - ID: x6uYqmx31kE)"}
    ]

    print("\nLoading audio samples...")
    embeddings = []
    for s in samples:
        waveform = load_audio(s["file"])
        embedding = get_embedding(model, waveform)
        embeddings.append(embedding)
        print(f"  Loaded {s['file']} -> Embedding size: {list(embedding.shape)}")

    # 5. Perform speaker verification tests
    print("\n" + "="*60)
    print("  Speaker Verification Results (Cosine Similarity)")
    print("="*60)
    
    # Cosine Similarity is simply the dot product since our embeddings are L2 normalized
    
    # Compare Sample 1 (Speaker A) and Sample 2 (Speaker B) - EXPECTED: Different speaker (low score)
    sim_1_2 = torch.dot(embeddings[0], embeddings[1]).item()
    
    # Compare Sample 2 (Speaker B) and Sample 3 (Speaker B) - EXPECTED: Same speaker (high score)
    sim_2_3 = torch.dot(embeddings[1], embeddings[2]).item()
    
    # Compare Sample 1 (Speaker A) and Sample 3 (Speaker B) - EXPECTED: Different speaker (low score)
    sim_1_3 = torch.dot(embeddings[0], embeddings[2]).item()

    print(f"Pair 1: {samples[0]['name']} vs {samples[1]['name']}")
    print(f"  -> Cosine Similarity: {sim_1_2:.4f}")
    print(f"  -> Match? {'YES' if sim_1_2 > 0.4 else 'NO'} (Threshold ~0.40)\n")

    print(f"Pair 2: {samples[1]['name']} vs {samples[2]['name']}")
    print(f"  -> Cosine Similarity: {sim_2_3:.4f}")
    print(f"  -> Match? {'YES' if sim_2_3 > 0.4 else 'NO'} (Threshold ~0.40)\n")

    print(f"Pair 3: {samples[0]['name']} vs {samples[2]['name']}")
    print(f"  -> Cosine Similarity: {sim_1_3:.4f}")
    print(f"  -> Match? {'YES' if sim_1_3 > 0.4 else 'NO'} (Threshold ~0.40)\n")

if __name__ == "__main__":
    main()
