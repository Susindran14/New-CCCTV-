# CrimeShield AI - Real-Time Surveillance System

A real-time violence detection system using YOLOv8 pose estimation and a tri-stream neural network (pose, motion, audio).

## Project Structure

```
RWF-2000/
├── app.py                          # Flask backend server
├── webcam_predict.py               # Original webcam prediction (CLI version)
├── model.py                        # TriStreamViolenceModel architecture
├── train.py                        # Model training script
├── predict.py                      # Prediction utilities
├── preprocess_dataset.py           # Feature extraction
├── best_tristream_model.pth        # Trained model weights
├── yolov8n-pose.pt                 # YOLOv8 pose model
├── templates/
│   └── ui.html                     # Web UI (served by Flask)
├── processed_features/             # Preprocessed features cache
│   ├── train/
│   ├── val/
│   └── test/
└── requirements.txt                # Python dependencies
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install Flask Flask-CORS torch torchvision numpy opencv-python opencv-contrib-python
```

### 2. Verify Model Files

Make sure these files exist in the project directory:
- `best_tristream_model.pth` - Trained violence detection model
- `yolov8n-pose.pt` - YOLOv8 nano pose model

## Usage

### Option 1: Web UI (Recommended)

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

Features:
- 🎥 Live video feed with real-time violence detection
- 📊 Real-time metrics (Pose, Motion, Audio scores)
- ⚠️ Instant alerts with confidence scores
- 🎬 Multi-camera support (UI ready)
- 📈 Event analysis panel
- 🔊 Audio analysis visualization

### Option 2: CLI (OpenCV Window)

```bash
python webcam_predict.py
```

Features:
- Displays live feed in OpenCV window
- Real-time confidence scores
- Press 'q' to quit

## System Architecture

### Backend (Flask)
- **VideoStreamManager**: Handles camera capture and frame buffering
- **Model Inference**: Runs TriStreamViolenceModel on background thread
- **RESTful APIs**: Serve video frames, detection data, and metrics

### Frontend (HTML/CSS/JS)
- **Real-time Video**: MJPEG stream from `/video_feed`
- **Live Metrics**: Fetches from `/api/detection-data` every 500ms
- **Status Updates**: Polls `/api/status` for camera health

### Model (PyTorch)
- **Input**: Video frames (640x480)
- **Features Extracted**:
  - **Pose**: YOLOv8 skeleton keypoints
  - **Motion**: Optical flow between frames
  - **Audio**: Mel-spectrogram or MFCC features
- **Output**: Violence confidence score [0.0, 1.0]

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI page |
| `/video_feed` | GET | MJPEG video stream |
| `/api/detection-data` | GET | Current detection metrics (JSON) |
| `/api/status` | GET | System status (JSON) |
| `/api/stream-metrics` | GET | Real-time metrics (Server-Sent Events) |

### Detection Data Response
```json
{
  "confidence": 0.52,
  "status": "WARNING",
  "alert": false,
  "timestamp": "2026-08-17T22:34:11.123456",
  "pose_score": 0.45,
  "motion_score": 0.62,
  "audio_score": 0.36,
  "event_id": "CS-170826-52"
}
```

## Configuration

Edit `app.py` to adjust:

```python
# Video capture settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # Frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     # Frame height
cap.set(cv2.CAP_PROP_FPS, 15)               # FPS

# Buffer settings
chunk_size = 16                              # Frames per prediction

# Detection thresholds
if avg_confidence >= 0.5:   # Violence threshold
if avg_confidence >= 0.35:  # Warning threshold
```

## Performance

- **GPU**: ~60-80 FPS (with CUDA)
- **CPU**: ~15-25 FPS
- **Memory**: ~2GB (model + buffers)
- **Latency**: ~1-2 seconds (batch processing every 16 frames @ 15 FPS)

## Troubleshooting

### Camera Not Found
```
❌ Error: Could not open webcam.
```
**Solution**: Ensure no other application is using the camera, or specify a different camera index:
```python
cap = cv2.VideoCapture(1)  # Try camera 1 instead of 0
```

### Model Weights Not Found
```
❌ Model weights file 'best_tristream_model.pth' not found!
```
**Solution**: Download or train the model and save it as `best_tristream_model.pth`

### Out of Memory
**Solution**: Reduce `chunk_size` or frame resolution in `app.py`

### Slow Detection
**Solution**: 
- Use GPU: Install CUDA and cuDNN
- Reduce frame resolution
- Increase buffer size for faster batching

## Training a New Model

For the audio stream to learn, the source videos must contain audio tracks.
The `archive/train` and `archive/test` XD-Violence splits contain audio. The
preprocessor maps folder `A` to `NonFight` and the event-category folders to
`Fight`. Process the splits separately to keep validation clips out of training:

```bash
python val/preprocess_dataset.py --data-dir archive/train --output-dir processed_features/train
python val/preprocess_dataset.py --data-dir archive/test --output-dir processed_features/val
python train.py --epochs 50
```

Preprocessing reports how many videos had usable audio. Training with audio
enabled stops if every training feature has silent audio. The separate RWF
videos in `train/` do not contain audio tracks, so they cannot train the
model's audio stream.

```bash
python train.py --epochs 50 --batch-size 32 --lr 0.001
```

See `train.py` for full options.

## Dataset Format

Expected structure for training:
```
processed_features/
├── train/
│   ├── Fight/
│   │   ├── video_id_0.npz  (contains 'pose', 'motion', 'audio')
│   │   └── ...
│   └── NonFight/
├── val/
│   ├── Fight/
│   └── NonFight/
└── test/
    ├── Fight/
    └── NonFight/
```

## License

MIT License - See LICENSE file

## Contact

For issues or questions, open a GitHub issue or contact the development team.
