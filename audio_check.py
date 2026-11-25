import os
import numpy as np
from pathlib import Path

def check_features_audio(features_dir="processed_features/val"):
    path = Path(features_dir)
    if not path.exists():
        print(f"❌ Error: Directory '{features_dir}' not found. Check your path!")
        return

    npz_files = list(path.glob("**/*.npz"))
    if not npz_files:
        print(f"❌ No .npz files found in {features_dir}")
        return

    print(f"🔍 Scanning {len(npz_files)} feature files for audio data...\n")

    zero_audio_count = int(0)
    active_audio_count = int(0)

    for file_path in npz_files[:50]:  # Check first 50 files as a sample
        data = np.load(file_path)
        audio_arr = data['audio']
        
        # Check if the array is entirely zeros or close to it
        is_zero = np.all(audio_arr == 0) or (np.abs(audio_arr).max() < 1e-8)
        
        if is_zero:
            zero_audio_count += 1
        else:
            active_audio_count += 1

    print("="*40)
    print(f"📊 Audio Inspection Results (Sample Size):")
    print(f"   - Files with ACTIVE Audio Data: {active_audio_count}")
    print(f"   - Files with ZERO/Silent Audio: {zero_audio_count}")
    print("="*40)

    if active_audio_count == 0:
        print("\n⚠️ WARNING: All sampled files have zero audio! The audio stream is being bypassed.")
    else:
        print("\n✅ SUCCESS: Non-zero audio Mel-spectrograms found! The audio stream is active.")

if __name__ == "__main__":
    check_features_audio()