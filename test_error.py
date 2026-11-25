import sys
import os
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VAL_DIR = ROOT_DIR / "val"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(VAL_DIR) not in sys.path:
    sys.path.insert(0, str(VAL_DIR))

try:
    print("Testing imports...")
    import torch
    print("✅ torch imported")
    
    import cv2
    print("✅ cv2 imported")
    
    from model import TriStreamViolenceModel
    print("✅ model imported")
    
    from val.preprocess_dataset import process_single_video
    print("✅ preprocess_dataset imported")
    
    print("\nTesting model initialization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TriStreamViolenceModel().to(device)
    model.eval()
    print("✅ Model initialized")
    
    print("\nTesting model loading...")
    model.load_state_dict(torch.load("best_tristream_model.pth", map_location=device))
    print("✅ Model weights loaded")
    
    print("\nTesting webcam access...")
    cap = cv2.VideoCapture(0)
    print(f"Webcam opened: {cap.isOpened()}")
    
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Can read frame: {ret}")
        if ret:
            print(f"Frame shape: {frame.shape}")
        cap.release()
        print("✅ Webcam works")
    else:
        print("⚠️ Webcam not available (this is OK on headless systems)")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
