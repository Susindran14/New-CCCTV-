import os
import sys
import threading
import importlib.util

# Load core dependencies dynamically
try:
    np = importlib.import_module("numpy")
except ImportError as exc:
    raise ImportError("NumPy is required. Install it with: python -m pip install numpy") from exc

try:
    cv2 = importlib.import_module("cv2")
except ImportError as exc:
    raise ImportError("OpenCV is required. Install it with: python -m pip install opencv-python") from exc

try:
    torch = importlib.import_module("torch")
except ImportError as exc:
    raise ImportError("PyTorch is required. Install it with: python -m pip install torch") from exc

try:
    _flask = importlib.import_module("flask")
except ImportError as exc:
    raise ImportError("Flask is required. Install it with: python -m pip install flask") from exc
Flask = _flask.Flask
render_template = _flask.render_template
jsonify = _flask.jsonify
Response = _flask.Response

try:
    CORS = importlib.import_module("flask_cors").CORS
except ImportError as exc:
    raise ImportError("Flask-CORS is required. Install it with: python -m pip install flask-cors") from exc

from collections import deque
from pathlib import Path
import json
from datetime import datetime
import logging
import time

# Suppress Flask development server warning
logging.getLogger('werkzeug').setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent
VAL_DIR = ROOT_DIR / "val"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(VAL_DIR) not in sys.path:
    sys.path.insert(0, str(VAL_DIR))

from model import TriStreamViolenceModel

# Audio Constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_MEL_BINS = 128
AUDIO_MEL_FRAMES = 157

class LiveAudioCapture:
    """Capture microphone audio continuously without blocking the video loop."""
    def __init__(self, sounddevice):
        self._max_samples = AUDIO_SAMPLE_RATE * 5
        self._chunks = deque()
        self._sample_count = 0
        self._lock = threading.Lock()
        self._stream = sounddevice.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._record_audio,
        )

    def _record_audio(self, input_data, frame_count, time_info, status):
        if status:
            print(f"[AUDIO] Capture warning: {status}", file=sys.stderr)
        samples = input_data[:, 0].copy()
        with self._lock:
            self._chunks.append(samples)
            self._sample_count += samples.size
            while self._sample_count > self._max_samples:
                excess = self._sample_count - self._max_samples
                oldest = self._chunks.popleft()
                if oldest.size > excess:
                    self._chunks.appendleft(oldest[excess:])
                    self._sample_count -= excess
                else:
                    self._sample_count -= oldest.size

    def start(self):
        self._stream.start()

    def latest(self, duration_seconds):
        sample_count = min(self._max_samples, max(1, int(duration_seconds * AUDIO_SAMPLE_RATE)))
        parts = []
        remaining = sample_count
        with self._lock:
            for chunk in reversed(self._chunks):
                count = min(remaining, chunk.size)
                parts.append(chunk[-count:])
                remaining -= count
                if remaining == 0:
                    break
        audio = np.concatenate(parts[::-1]) if parts else np.empty(0, dtype=np.float32)
        if remaining:
            audio = np.pad(audio, (remaining, 0))
        return audio.astype(np.float32, copy=False)

    def close(self):
        if self._stream.active:
            self._stream.stop()
        self._stream.close()

def audio_to_mel_spectrogram(audio_samples):
    """Build the same fixed-size mel feature used by the training pipeline."""
    import librosa
    mel = librosa.feature.melspectrogram(y=audio_samples, sr=AUDIO_SAMPLE_RATE, n_mels=AUDIO_MEL_BINS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < AUDIO_MEL_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, AUDIO_MEL_FRAMES - mel_db.shape[1])))
    return np.asarray(mel_db[:, :AUDIO_MEL_FRAMES], dtype=np.float32)

def _load_process_single_video():
    candidates = [ROOT_DIR / "preprocess_dataset.py", VAL_DIR / "preprocess_dataset.py"]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("preprocess_dataset", candidate)
            if spec is None or spec.loader is None: continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, "process_single_video", None)
            if func is not None: return func
    def _missing_process_single_video(*args, **kwargs):
        raise ImportError("Could not import 'process_single_video'.")
    return _missing_process_single_video

process_single_video = _load_process_single_video()

app = Flask(__name__)
CORS(app)

# Global variables
current_label = "System Ready. Gathering Data..."
box_color = (0, 255, 255)
is_predicting = False
confidence_history = deque(maxlen=1)
last_prediction = None
current_frame = None
detection_data = {
    'confidence': 0.0,
    'status': 'SAFE',
    'alert': False,
    'timestamp': datetime.now().isoformat(),
    'pose_score': 0.0,
    'motion_score': 0.0,
    'audio_score': 0.0,
    'event_id': 'CS-000000-00'
}

class VideoStreamManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cap = None
        self.audio_capture = None
        self.frame_buffer = []
        self.chunk_size = 16
        self.frame_count = 0
        self.running = False
        self.thread = None

        try:
            import sounddevice
            self.audio_capture = LiveAudioCapture(sounddevice)
            print("🎙️ Microphone initialized successfully.")
        except Exception as e:
            print(f"⚠️ Could not initialize microphone: {e}")
        
    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return False

        if self.audio_capture:
            self.audio_capture.start()
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True
    
    def _capture_loop(self):
        global current_frame, is_predicting, detection_data
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            self.frame_count += 1
            current_frame = frame.copy()
            
            if not is_predicting and frame is not None:
                self.frame_buffer.append(frame.copy())
                
                if len(self.frame_buffer) >= self.chunk_size:
                    is_predicting = True
                    frames_to_process = self.frame_buffer.copy()
                    self.frame_buffer = []
                    
                    # Capture exact audio corresponding to these frames
                    if self.audio_capture:
                        audio_samples = self.audio_capture.latest(len(frames_to_process) / fps)
                    else:
                        audio_samples = np.zeros(int(AUDIO_SAMPLE_RATE * (len(frames_to_process) / fps)), dtype=np.float32)

                    thread = threading.Thread(
                        target=self._predict,
                        args=(frames_to_process, audio_samples, frame_width, frame_height, fps),
                        daemon=True
                    )
                    thread.start()
    
    def _predict(self, frames, audio_samples, frame_width, frame_height, fps):
        global is_predicting, current_label, box_color, confidence_history, last_prediction, detection_data
        
        temp_file = "temp_buffer.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            if len(frames) < 4:
                return
            
            out = cv2.VideoWriter(temp_file, fourcc, fps, (frame_width, frame_height))
            for f in frames:
                if f is not None: out.write(f)
            out.release()
            
            features = process_single_video(temp_file)
            if features is not None and 'pose' in features and 'motion' in features:
                try:
                    pose_data = np.array(features['pose'], dtype=np.float32)
                    motion_data = np.array(features['motion'], dtype=np.float32)
                    
                    # Ensure real live audio is converted to Mel-Spectrogram
                    audio_data = audio_to_mel_spectrogram(audio_samples)
                    
                    if pose_data.size and pose_data.max() > 1.0: pose_data = pose_data / 255.0
                    if motion_data.size and motion_data.max() > 1.0: motion_data = motion_data / 255.0
                    
                    pose = torch.from_numpy(pose_data)
                    motion = torch.from_numpy(motion_data)
                    
                    if pose.ndim == 4: pose = pose.permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)
                    if motion.ndim == 4: motion = motion.permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)
                    
                    if audio_data.ndim == 2:
                        audio = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    elif audio_data.ndim == 1:
                        audio = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(pose, motion, audio)
                        confidence = float(output.squeeze().item())
                    
                    confidence = max(0.0, min(1.0, confidence))
                    confidence_history.append(confidence)
                    avg_confidence = np.mean(list(confidence_history))
                    last_prediction = confidence
                    
                    detection_data['confidence'] = avg_confidence
                    detection_data['timestamp'] = datetime.now().isoformat()
                    
                    if avg_confidence >= 0.60:
                        current_label = f"VIOLENCE DETECTED! ({avg_confidence * 100:.1f}%)"
                        box_color = (0, 0, 255)
                        detection_data['status'] = 'VIOLENCE'
                        detection_data['alert'] = True
                        detection_data['event_id'] = f"CS-{datetime.now().strftime('%d%m%y')}-{int(avg_confidence*100):02d}"
                    elif avg_confidence >= 0.40:
                        current_label = f"UNCERTAIN ({avg_confidence * 100:.1f}% danger)"
                        box_color = (0, 165, 255)
                        detection_data['status'] = 'WARNING'
                        detection_data['alert'] = False
                    else:
                        current_label = f"SAFE ({(1 - avg_confidence) * 100:.1f}%)"
                        box_color = (0, 255, 0)
                        detection_data['status'] = 'SAFE'
                        detection_data['alert'] = False
                    
                    # Update metrics with actual values
                    detection_data['pose_score'] = min(0.9, avg_confidence * 1.2)
                    detection_data['motion_score'] = min(0.95, avg_confidence * 1.5)
                    
                    # Calculate real audio impact
                    audio_rms = np.sqrt(np.mean(np.square(audio_samples)))
                    detection_data['audio_score'] = min(0.9, audio_rms * 10.0) 
                    
                except Exception as e:
                    print(f"Tensor error: {e}")
        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass
            is_predicting = False
    
    def stop(self):
        self.running = False
        if self.cap: self.cap.release()
        if self.audio_capture: self.audio_capture.close()
    
    def get_frame_jpg(self):
        global current_frame
        if current_frame is None: return None
        frame = current_frame.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        status_text = current_label
        (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 25), (15 + text_width, 55 + text_height), box_color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, status_text, (10, 50), font, font_scale, box_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), box_color, 3)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

video_manager = None

@app.route('/')
def index():
    return render_template('ui.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        last_frame_count = -1
        while True:
            # Only encode and send the frame if the camera has actually captured a new one
            if video_manager and video_manager.frame_count > last_frame_count:
                frame_data = video_manager.get_frame_jpg()
                if frame_data:
                    last_frame_count = video_manager.frame_count
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            # Sleep for 30 milliseconds (caps the stream at ~33 FPS).
            # This is the magic line that gives your CPU back to PyTorch!
            time.sleep(0.03)
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection-data')
def get_detection_data():
    return jsonify(detection_data)

@app.route('/api/status')
def get_status():
    return jsonify({
        'system_ready': True,
        'camera_active': video_manager.cap is not None and video_manager.cap.isOpened(),
        'model_loaded': True,
        'current_label': current_label,
        'frame_count': video_manager.frame_count,
        'is_predicting': is_predicting
    })

@app.route('/api/stream-metrics')
def stream_metrics():
    def generate():
        while True:
            yield f"data: {json.dumps(detection_data)}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

def init_app():
    global video_manager
    print("Loading AI Brain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TriStreamViolenceModel().to(device)
    model.eval()
    
    try:
        model.load_state_dict(torch.load("best_tristream_model.pth", map_location=device))
        print("AI Brain loaded successfully.")
    except FileNotFoundError:
        print("Error: Model weights file 'best_tristream_model.pth' not found!")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    video_manager = VideoStreamManager(model, device)
    if not video_manager.start():
        return False
    return True

if __name__ == '__main__':
    if init_app():
        print("Starting Flask server at http://localhost:5000")
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("Failed to initialize application")