import os
import sys
import threading
import numpy as np
from collections import deque
from pathlib import Path

import cv2
import torch

ROOT_DIR = Path(__file__).resolve().parent
VAL_DIR = ROOT_DIR / "val"
AUDIO_SAMPLE_RATE = 16000
AUDIO_MEL_BINS = 128
AUDIO_MEL_FRAMES = 157

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(VAL_DIR) not in sys.path:
    sys.path.insert(0, str(VAL_DIR))

from model import TriStreamViolenceModel

try:
    from preprocess_dataset import process_single_video  # type: ignore
except ImportError:
    import importlib.util

    module_path = ROOT_DIR / "preprocess_dataset.py"
    if not module_path.exists():
        raise

    spec = importlib.util.spec_from_file_location("preprocess_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preprocess_dataset from {module_path}")

    preprocess_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_dataset)
    process_single_video = preprocess_dataset.process_single_video

# Global variables so the background thread can update the live screen
current_label = "System Ready. Gathering Data..."
box_color = (0, 255, 255)
is_predicting = False

# ---------------------------------------------------------
# FIX: Reduced moving average lag from 10 to 3 frames
# This stops old audio noises from dragging the score up
# ---------------------------------------------------------
confidence_history = deque(maxlen=1) 
last_prediction = None


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
        sample_count = min(
            self._max_samples,
            max(1, int(duration_seconds * AUDIO_SAMPLE_RATE)),
        )
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

    mel = librosa.feature.melspectrogram(
        y=audio_samples,
        sr=AUDIO_SAMPLE_RATE,
        n_mels=AUDIO_MEL_BINS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < AUDIO_MEL_FRAMES:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, AUDIO_MEL_FRAMES - mel_db.shape[1])),
        )
    return np.asarray(mel_db[:, :AUDIO_MEL_FRAMES], dtype=np.float32)


def background_prediction(frames, audio_samples, frame_width, frame_height, fps, device, model):
    """Runs in a background thread so the webcam feed stays responsive."""
    global current_label, box_color, is_predicting, confidence_history, last_prediction

    temp_file = "temp_buffer.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        # Ensure frames are properly formatted
        if len(frames) < 4:
            current_label = "Not enough frames to judge yet"
            box_color = (0, 255, 255)
            return

        out = cv2.VideoWriter(temp_file, fourcc, fps, (frame_width, frame_height))
        for f in frames:
            if f is not None:
                out.write(f)
        out.release()

        features = process_single_video(temp_file)
        if features is not None and 'pose' in features and 'motion' in features:
            
            try:
                # Proper tensor conversion with correct normalization and shapes
                pose_data = np.array(features['pose'], dtype=np.float32)   # (T, H, W, C)
                motion_data = np.array(features['motion'], dtype=np.float32)  # (T, H, W, C)
                audio_data = audio_to_mel_spectrogram(audio_samples)

                print(f"[DEBUG] Feature shapes - Pose: {pose_data.shape}, Motion: {motion_data.shape}, Audio: {audio_data.shape}")
                print(f"[DEBUG] Pose range: [{pose_data.min():.2f}, {pose_data.max():.2f}], Motion range: [{motion_data.min():.2f}, {motion_data.max():.2f}]")
                print(f"[AUDIO] Microphone RMS: {np.sqrt(np.mean(np.square(audio_samples))):.6f}")

                # Normalize images to [0,1]
                if pose_data.size and pose_data.max() > 1.0:
                    pose_data = pose_data / 255.0
                if motion_data.size and motion_data.max() > 1.0:
                    motion_data = motion_data / 255.0

                # Convert to tensors and reshape to match model expectations:
                # Pose/Motion expected: (B, C=3, D=num_frames, H=224, W=224)
                # Audio expected: (B, C=1, H, W) for Conv2d
                pose = torch.from_numpy(pose_data)
                motion = torch.from_numpy(motion_data)

                if pose.ndim == 4:
                    # (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
                    pose = pose.permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
                else:
                    raise ValueError(f"Unexpected pose shape: {pose.shape}")

                if motion.ndim == 4:
                    motion = motion.permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
                else:
                    raise ValueError(f"Unexpected motion shape: {motion.shape}")

                if audio_data.ndim == 2:
                    audio = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).float().to(device)
                elif audio_data.ndim == 1:
                    audio = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).float().to(device)
                else:
                    raise ValueError(f"Unexpected audio shape: {audio_data.shape}")

                with torch.no_grad():
                    output = model(pose, motion, audio)
                    confidence = float(output.squeeze().item())

                print(f"[DEBUG] Model output shape: {output.shape}, Raw output: {output}, Confidence: {confidence:.4f}")

                # Clamp confidence to [0, 1] range
                confidence = max(0.0, min(1.0, confidence))
                confidence_history.append(confidence)
                
                # Use mean for robust prediction
                avg_confidence = np.mean(list(confidence_history))
                
                print(f"[DEBUG] Raw confidence: {confidence:.4f} | Avg: {avg_confidence:.4f} | History: {list(confidence_history)}")

                last_prediction = confidence
                
                # Adjusted thresholds for noisy room environments and OpenCV text rendering
                if avg_confidence >= 0.65:
                    current_label = f"VIOLENCE DETECTED! ({avg_confidence * 100:.1f}%)"
                    box_color = (0, 0, 255)  # Red
                elif avg_confidence >= 0.45:
                    current_label = f"UNCERTAIN ({avg_confidence * 100:.1f}% danger)"
                    box_color = (0, 165, 255)  # Orange
                else:
                    current_label = f"SAFE ({(1 - avg_confidence) * 100:.1f}%)"
                    box_color = (0, 255, 0)  # Green
            except Exception as tensor_error:
                print(f"Tensor processing error: {tensor_error}")
                current_label = "Processing error (shape mismatch)"
                box_color = (0, 255, 255)
        else:
            current_label = "Feature extraction failed"
            box_color = (0, 255, 255)
            print(f"[ERROR] Feature extraction returned None or missing keys. Features: {features}")
    except Exception as e:
        print(f"Background AI processing error: {e}")
        current_label = "Processing issue, retrying..."
        box_color = (0, 255, 255)
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        is_predicting = False


def main():
    global is_predicting, current_label, box_color

    print("Loading AI Brain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = TriStreamViolenceModel().to(device)
    model.eval()

    try:
        model.load_state_dict(torch.load("best_tristream_model.pth", map_location=device))
        print("AI Brain loaded successfully.")
    except FileNotFoundError:
        print("Error: Model weights file 'best_tristream_model.pth' not found in current directory!")
        print(f"   Current directory: {os.getcwd()}")
        return
    except Exception as e:
        print("Error loading model:", e)
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    audio_capture = None
    try:
        import sounddevice

        audio_capture = LiveAudioCapture(sounddevice)
        audio_capture.start()
    except Exception as e:
        if audio_capture is not None:
            audio_capture.close()
        print(f"Could not start microphone capture: {e}")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_buffer = []
    chunk_size = 16
    frame_count = 0

    print("Microphone capture active at 16 kHz.")
    print("Starting Live Feed (Optimized)... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Create a copy for display
            display_frame = frame.copy()

            # Draw border in current color (do this first)
            cv2.rectangle(display_frame, (0, 0), (frame_width - 1, frame_height - 1), box_color, 4)

            # Draw background for text for better visibility
            text = current_label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw semi-transparent dark background for text (larger area)
            margin = 10
            overlay = display_frame.copy()
            text_y_start = 15
            cv2.rectangle(overlay, (5, text_y_start), (text_width + 25, text_y_start + text_height + 20), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            # Draw text on top with dynamic color matching the status
            text_y_pos = text_y_start + text_height + 8
            cv2.putText(display_frame, text, (15, text_y_pos), font, font_scale, box_color, thickness, cv2.LINE_AA)

            # Draw frame counter (bottom-left)
            counter_text = f"Frame: {frame_count}"
            cv2.putText(display_frame, counter_text, (10, frame_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow('AI Violence Detection - LIVE', display_frame)

            # Add frame to buffer if not currently predicting
            if not is_predicting and frame is not None:
                frame_buffer.append(frame.copy())

                if len(frame_buffer) >= chunk_size:
                    is_predicting = True
                    frames_to_process = frame_buffer.copy()
                    frame_buffer = []
                    audio_samples = audio_capture.latest(len(frames_to_process) / fps)

                    print(f"[INFO] Processing buffer of {len(frames_to_process)} frames...")

                    ai_thread = threading.Thread(
                        target=background_prediction,
                        args=(frames_to_process, audio_samples, frame_width, frame_height, fps, device, model),
                        daemon=True
                    )
                    ai_thread.start()

            # Quit on 'q' - using 30ms delay for better window responsiveness
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC to quit
                break

    finally:
        audio_capture.close()
        cap.release()
        cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()