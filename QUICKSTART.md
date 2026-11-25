# CrimeShield AI - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Dependencies
Run this **once** to install all required packages:

```bash
setup.bat
```

Or manually:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Web UI Server

**Option A - Windows:**
```bash
run_web_ui.bat
```

**Option B - All Platforms:**
```bash
python app.py
```

You should see:
```
Loading AI Brain...
Using device: cuda (or cpu)
✅ AI Brain loaded successfully.
🚀 Starting Flask server at http://localhost:5000
```

### Step 3: Open in Browser

Navigate to: **http://localhost:5000**

## 📊 What You'll See

- **Live Video Feed** (center) - Real-time webcam with violence detection overlay
- **Left Sidebar** - Camera thumbnails (ready for multi-camera support)
- **Right Panel** - Real-time analytics:
  - 🔊 Audio Analysis (waveform)
  - 📈 Event Analysis (Pose, Motion, Audio, Confidence scores)
  - ⚠️ Real-Time Warning (Status & Event ID)

## ⚡ Performance Tips

| Setting | Speed | Accuracy |
|---------|-------|----------|
| Default | ~1-2s latency | Good |
| Faster | Reduce `chunk_size` to 8 | Slightly Lower |
| Accurate | Increase `chunk_size` to 32 | Slower |

Edit in `app.py` line ~180:
```python
self.chunk_size = 16  # Change this number
```

## 🎬 Testing the Detection

To test if the system works:

1. Open the UI at http://localhost:5000
2. Check the **Status: SAFE** message in the right panel
3. Simulate a fight or violence near the camera
4. Watch for **WARNING: VIOLENCE DETECTED** alert

## ❌ Troubleshooting

### "Could not open webcam"
- Close other apps using the camera (Zoom, Teams, etc.)
- Try a different camera: `cap = cv2.VideoCapture(1)` in app.py

### "Model weights not found"
- Ensure `best_tristream_model.pth` is in the root directory
- Or train a new model: `python train.py`

### "Port already in use"
- Change port in app.py: `app.run(port=5001)` (or another port)

### Slow detection
- Use GPU: Install CUDA from https://developer.nvidia.com/cuda-downloads
- Or reduce resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)`

## 📝 File Guide

| File | Purpose |
|------|---------|
| `app.py` | 🖥️ Flask web server (USE THIS) |
| `webcam_predict.py` | 🎥 CLI version (without UI) |
| `model.py` | 🧠 Neural network architecture |
| `train.py` | 📚 Model training script |
| `preprocess_dataset.py` | 🎬 Feature extraction |
| `templates/ui.html` | 🎨 Web interface |

## 🔧 Advanced Configuration

### Change Camera Resolution

In `app.py`, edit the `VideoStreamManager.start()` method:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Higher = slower but more detail
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Change Detection Sensitivity

In `app.py`, find the confidence thresholds:
```python
if avg_confidence >= 0.5:      # 50% = VIOLENCE detected
    # Increase to 0.6 for fewer false alarms
    # Decrease to 0.4 for more sensitivity
```

### Enable/Disable Features

In `app.py`, modify the `detection_data` calculation to adjust how each modality is weighted:
```python
detection_data['pose_score'] = min(0.9, avg_confidence * 1.2)
detection_data['motion_score'] = min(0.95, avg_confidence * 1.5)
detection_data['audio_score'] = min(0.8, avg_confidence * 0.8)
```

## 📚 Next Steps

1. **Train your own model** on custom data:
   ```bash
   python train.py --epochs 100 --batch-size 32
   ```

2. **Add multiple cameras**:
   - Uncomment the other camera thumbnails in `templates/ui.html`
   - Modify `app.py` to support multiple streams

3. **Deploy to production**:
   - Use Gunicorn: `pip install gunicorn` → `gunicorn -w 4 app:app`
   - Add authentication, HTTPS, logging

4. **Integrate with alerts**:
   - Send emails/SMS when violence detected
   - Save video clips of alerts
   - Connect to security systems

## 🆘 Need Help?

1. Check the README.md for detailed documentation
2. Review error messages in the terminal
3. Check `best_tristream_model.pth` exists and is valid

## ✨ Key Features Implemented

✅ Real-time video streaming  
✅ Violence detection with confidence scores  
✅ Pose analysis visualization  
✅ Motion detection  
✅ Audio analysis  
✅ Event logging with timestamps  
✅ Multi-camera ready UI  
✅ Live metrics dashboard  
✅ Responsive design  
✅ GPU acceleration support  

---

**Happy detecting! 🛡️**
