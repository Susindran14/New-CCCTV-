import sys
from pathlib import Path
import torch
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
VAL_DIR = ROOT_DIR / "val"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(VAL_DIR) not in sys.path:
    sys.path.insert(0, str(VAL_DIR))

from model import TriStreamViolenceModel
from preprocess_dataset import process_single_video

def predict_video(video_path, model_path="best_tristream_model.pth"):
    print(f"\nAnalyzing Video: {video_path}")
    
    # 1. Load the pre-trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriStreamViolenceModel().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ AI Brain loaded successfully.")
    except Exception as e:
        print("❌ Error: Could not find trained model. Wait for train.py to finish!")
        return

    # 2. Extract features from the new video
    print("Extracting Pose, Motion, and Audio streams...")
    features = process_single_video(video_path)
    
    if features is None:
        print("❌ Error: Video is too short or corrupted.")
        return

    # 3. Format data for the AI model
    # Add batch dimensions so the AI can read it properly
    pose = torch.tensor(features['pose'], dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    motion = torch.tensor(features['motion'], dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    audio = torch.tensor(features['audio'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pose, motion, audio = pose.to(device), motion.to(device), audio.to(device)

    # 4. Make the Prediction
    with torch.no_grad():
        output = model(pose, motion, audio)
        confidence = output.item()
        
    print("\n" + "="*40)
    if confidence >= 0.95:
        print(f"🚨 ALERT: VIOLENCE DETECTED! ")
        print(f"Confidence Score: {confidence*100:.2f}%")
    else:
        print(f"✅ Safe: No Violence.")
        print(f"Confidence Score: {(1-confidence)*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # We will change this name later when you download a test video!
    TEST_VIDEO_FILE = "test_video.mp4" 
    predict_video(TEST_VIDEO_FILE)