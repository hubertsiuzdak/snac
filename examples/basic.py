"""
Basic SNAC example: encode and decode audio

Download sample audio:
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O audio.wav

Run with:
uv run python examples/basic.py
"""

import torch
import soundfile as sf
from snac import SNAC

# Load the pretrained SNAC model
# Use snac_24khz for speech, snac_32khz or snac_44khz for music
print("Loading SNAC model...")
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

# Load audio file
print("Loading audio...")
audio, sr = sf.read("audio.wav", dtype='float32')

# Convert to mono if stereo
if audio.ndim > 1:
    print("Converting to mono...")
    audio = audio.mean(axis=1)

# Resample if necessary (SNAC 24khz expects 24kHz audio)
if sr != 24000:
    print(f"Resampling from {sr}Hz to 24000Hz...")
    # Simple resampling using numpy
    from scipy import signal
    num_samples = int(len(audio) * 24000 / sr)
    audio = signal.resample(audio, num_samples)
    sr = 24000

# Convert to torch tensor and add channel dimension
audio = torch.from_numpy(audio).unsqueeze(0)

# Add batch dimension and move to device
audio = audio.unsqueeze(0).to(device)  # Shape: (1, 1, T)

# Encode audio to codes
print("Encoding audio...")
with torch.inference_mode():
    codes = model.encode(audio)

print(f"Encoded to {len(codes)} code levels with shapes: {[code.shape for code in codes]}")

# Decode codes back to audio
print("Decoding audio...")
with torch.inference_mode():
    audio_reconstructed = model.decode(codes)

# Save reconstructed audio
print("Saving reconstructed audio...")
audio_reconstructed = audio_reconstructed.squeeze().cpu().numpy()
sf.write("audio_reconstructed.wav", audio_reconstructed, sr)

print("✓ Done! Reconstructed audio saved to audio_reconstructed.wav")