import os
import glob
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TriStreamViolenceModel

class PreprocessedViolenceDataset(Dataset):
    def __init__(self, feature_dir, require_audio=False):
        self.file_paths = []
        self.labels = []
        active_audio_count = 0
        silent_audio_count = 0

        fight_files = glob.glob(os.path.join(feature_dir, "Fight", "*.npz"))
        for f in fight_files:
            self.file_paths.append(f)
            self.labels.append(1.0)

        nonfight_files = glob.glob(os.path.join(feature_dir, "NonFight", "*.npz"))
        for f in nonfight_files:
            self.file_paths.append(f)
            self.labels.append(0.0)

        for file_path in self.file_paths:
            with np.load(file_path) as data:
                audio = data["audio"]
                if np.any(np.abs(audio) > 1e-8):
                    active_audio_count += 1
                else:
                    silent_audio_count += 1

        print(f"Dataset Loaded: {len(self.file_paths)} total samples from {feature_dir}")
        print(
            f"Audio features: {active_audio_count} active, "
            f"{silent_audio_count} silent."
        )
        if require_audio and self.file_paths and active_audio_count == 0:
            raise ValueError(
                f"No active audio features found in {feature_dir}. "
                "Use videos that contain audio and regenerate the features before "
                "training with the audio stream enabled."
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        
        pose = torch.tensor(data['pose'], dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        motion = torch.tensor(data['motion'], dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        audio = torch.tensor(data['audio'], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        return pose, motion, audio, label

def evaluate_model(model, data_loader, criterion, device, use_pose=True, use_motion=True, use_audio=True):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        validation_progress = tqdm(data_loader, desc="Validation", unit="batch")
        for pose, motion, audio, label in validation_progress:
            pose = pose.to(device, non_blocking=non_blocking)
            motion = motion.to(device, non_blocking=non_blocking)
            audio = audio.to(device, non_blocking=non_blocking)
            label = label.to(device, non_blocking=non_blocking)

            outputs = model(pose, motion, audio, use_pose=use_pose, use_motion=use_motion, use_audio=use_audio)
            loss = criterion(outputs, label)
            running_loss += loss.item() * label.size(0)
            predictions = (outputs >= 0.5).float()
            correct_predictions += (predictions == label).sum().item()
            total_samples += label.size(0)
            validation_progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        return 0.0, 0.0
    return running_loss / total_samples, (correct_predictions / total_samples) * 100.0

def evaluate_accuracy(model, data_loader, device, use_pose=True, use_motion=True, use_audio=True):
    _, accuracy = evaluate_model(
        model, data_loader, nn.BCELoss(), device,
        use_pose=use_pose, use_motion=use_motion, use_audio=use_audio
    )
    return accuracy

def train_tristream_model(epochs=50, lr=1e-4, weight_decay=1e-4, patience=8,
                          use_pose=True, use_motion=True, use_audio=True, resume=False):
    train_dir = "processed_features/train"
    val_dir = "processed_features/val"
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist yet! Wait for preprocessing to complete.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")
    if pin_memory:
        print(f"CUDA GPU: {torch.cuda.get_device_name(device)}")

    dataset = PreprocessedViolenceDataset(train_dir, require_audio=use_audio)
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=pin_memory
    )

    val_loader = None
    if os.path.exists(val_dir):
        val_dataset = PreprocessedViolenceDataset(val_dir, require_audio=use_audio)
        val_loader = DataLoader(
            val_dataset, batch_size=4, shuffle=False, num_workers=0,
            pin_memory=pin_memory,
        )
    else:
        print(f"Warning: {val_dir} does not exist; validation accuracy will not be calculated.")

    model = TriStreamViolenceModel(
        use_pose=use_pose, use_motion=use_motion, use_audio=use_audio
    ).to(device)

    # ---------------------------------------------------------
    # RESUME TRAINING LOGIC
    # ---------------------------------------------------------
    if resume and os.path.exists("best_tristream_model.pth"):
        print("\n📥 Resuming training from previously saved best_tristream_model.pth...")
        model.load_state_dict(torch.load("best_tristream_model.pth", map_location=device))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

    early_stopping_patience = patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_path = "training_history.csv"

    # Only create a new history file if we are NOT resuming
    if not resume:
        with open(history_path, "w", newline="") as history_file:
            csv.writer(history_file).writerow([
                "epoch", "learning_rate", "train_loss", "train_accuracy",
                "val_loss", "val_accuracy"
            ])
            
    print("\nStarting Training Loop...")
    print(f"Total batches to process: {len(train_loader)}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        for pose, motion, audio, label in train_progress:
            pose = pose.to(device, non_blocking=pin_memory)
            motion = motion.to(device, non_blocking=pin_memory)
            audio = audio.to(device, non_blocking=pin_memory)
            label = label.to(device, non_blocking=pin_memory)

            # ---------------------------------------------------------
            # MODALITY DROPOUT: Fix Visual Dominance for IEEE Paper
            # ---------------------------------------------------------
            if use_audio and torch.rand(1).item() < 0.20:
                pose = torch.zeros_like(pose)
                motion = torch.zeros_like(motion)

            optimizer.zero_grad()
            outputs = model(pose, motion, audio, use_pose=use_pose, use_motion=use_motion, use_audio=use_audio)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * pose.size(0)
            predictions = (outputs >= 0.5).float()
            correct_predictions += (predictions == label).sum().item()
            total_samples += label.size(0)
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_predictions / total_samples) * 100.0

        val_loss = float("nan")
        val_acc = float("nan")
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(
                model, val_loader, criterion, device,
                use_pose=use_pose, use_motion=use_motion, use_audio=use_audio
            )
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - "
            f"Accuracy: {epoch_acc:.2f}% - Val Loss: {val_loss:.4f} - "
            f"Val Accuracy: {val_acc:.2f}% - LR: {current_lr:.6g}"
        )

        with open(history_path, "a", newline="") as history_file:
            csv.writer(history_file).writerow([
                epoch + 1, current_lr, epoch_loss, epoch_acc, val_loss, val_acc
            ])

        if val_loader is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), "best_tristream_model.pth")
                print(f" -> Validation loss improved to {val_loss:.4f}. Model weights saved to best_tristream_model.pth")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered: validation loss has not improved for {early_stopping_patience} consecutive epochs.")
                    break

    model_save_path = "best_tristream_model.pth"
    if val_loader is None:
        torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining Complete! Model weights saved as {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tri-Stream Violence Detection Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--use_pose", action="store_true", default=True, help="Enable pose stream")
    parser.add_argument("--no_pose", dest="use_pose", action="store_false")
    parser.add_argument("--use_motion", action="store_true", default=True, help="Enable motion stream")
    parser.add_argument("--no_motion", dest="use_motion", action="store_false")
    parser.add_argument("--use_audio", action="store_true", default=True, help="Enable audio stream")
    parser.add_argument("--no_audio", dest="use_audio", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Resume from saved model weights") # <--- Added here
    
    args = parser.parse_args()

    train_tristream_model(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_pose=args.use_pose,
        use_motion=args.use_motion,
        use_audio=args.use_audio,
        resume=args.resume # <--- Passed here
    )