import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import TriStreamViolenceModel
from train import PreprocessedViolenceDataset

def evaluate_ieee_metrics(model, data_loader, device, use_pose=True, use_motion=True, use_audio=True, simulate_occlusion=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pose, motion, audio, label in data_loader:
            pose = pose.to(device)
            motion = motion.to(device)
            audio = audio.to(device)
            label = label.to(device)

            # ---------------------------------------------------------
            # SIMULATE CAMERA BLOCKAGE FOR IEEE ROBUSTNESS TEST
            # ---------------------------------------------------------
            if simulate_occlusion:
                pose = torch.zeros_like(pose)
                motion = torch.zeros_like(motion)

            outputs = model(pose, motion, audio, use_pose=use_pose, use_motion=use_motion, use_audio=use_audio)
            preds = (outputs >= 0.5).float()

            all_preds.extend(preds.detach().cpu().numpy().flatten())
            all_labels.extend(label.detach().cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = np.sum((all_labels == 1) & (all_preds == 1))
    tn = np.sum((all_labels == 0) & (all_preds == 0))
    fp = np.sum((all_labels == 0) & (all_preds == 1))
    fn = np.sum((all_labels == 1) & (all_preds == 0))

    confusion_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]

    total = len(all_labels)
    accuracy = ((tp + tn) / total * 100.0) if total > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Sensitivity
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(total)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tri-Stream Violence Model on IEEE Standard Metrics")
    parser.add_argument(
        "--mode",
        type=str,
        default="tristream",
        choices=[
            "tristream", "tri-stream",
            "dualstream", "dual-stream", "pose-motion",
            "pose-only", "pose",
            "motion-only", "motion",
            "audio-only", "audio"
        ],
        help="Stream ablation mode to evaluate"
    )
    parser.add_argument("--feature_dir", type=str, default="processed_features/val", help="Path to validation features")
    parser.add_argument("--model_path", type=str, default="best_tristream_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--use_pose", action="store_true", default=None, help="Explicitly enable pose stream")
    parser.add_argument("--no_pose", dest="use_pose", action="store_false", help="Explicitly disable pose stream")
    parser.add_argument("--use_motion", action="store_true", default=None, help="Explicitly enable motion stream")
    parser.add_argument("--no_motion", dest="use_motion", action="store_false", help="Explicitly disable motion stream")
    parser.add_argument("--use_audio", action="store_true", default=None, help="Explicitly enable audio stream")
    parser.add_argument("--no_audio", dest="use_audio", action="store_false", help="Explicitly disable audio stream")
    
    # New argument to test visual occlusion robustness
    parser.add_argument("--simulate_occlusion", action="store_true", help="Zero out visual tensors to test audio robustness (Camera Blocked)")

    args = parser.parse_args()

    # Determine stream configuration from mode
    mode = args.mode.lower().replace("_", "-")
    if mode in ["tristream", "tri-stream"]:
        use_pose, use_motion, use_audio = True, True, True
    elif mode in ["dualstream", "dual-stream", "pose-motion"]:
        use_pose, use_motion, use_audio = True, True, False
    elif mode in ["pose-only", "pose"]:
        use_pose, use_motion, use_audio = True, False, False
    elif mode in ["motion-only", "motion"]:
        use_pose, use_motion, use_audio = False, True, False
    elif mode in ["audio-only", "audio"]:
        use_pose, use_motion, use_audio = False, False, True
    else:
        use_pose, use_motion, use_audio = True, True, True

    # Override if explicit boolean flags were set
    if args.use_pose is not None:
        use_pose = args.use_pose
    if args.use_motion is not None:
        use_motion = args.use_motion
    if args.use_audio is not None:
        use_audio = args.use_audio

    if not os.path.isdir(args.feature_dir):
        raise FileNotFoundError(f"Validation features directory not found: {args.feature_dir}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model weights file not found: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PreprocessedViolenceDataset(args.feature_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No .npz feature files found in {args.feature_dir}")

    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = TriStreamViolenceModel(
        use_pose=use_pose, use_motion=use_motion, use_audio=use_audio
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Pass the occlusion flag to the metrics evaluator
    metrics = evaluate_ieee_metrics(
        model, data_loader, device,
        use_pose=use_pose, use_motion=use_motion, use_audio=use_audio,
        simulate_occlusion=args.simulate_occlusion
    )

    print("\n" + "=" * 65)
    print(" Tri-Stream Violence Detection - IEEE Standard Evaluation")
    print("=" * 65)
    print(f" Ablation Mode   : {args.mode}")
    print(f" Streams Active  : Pose={use_pose}, Motion={use_motion}, Audio={use_audio}")
    print(f" Camera Occluded : {args.simulate_occlusion}")
    print(f" Total Samples   : {metrics['total']}")
    print("-" * 65)
    print(f" Accuracy (%)         : {metrics['accuracy']:.2f}%")
    print(f" Precision            : {metrics['precision']:.4f}")
    print(f" Recall (Sensitivity) : {metrics['recall']:.4f}")
    print(f" Specificity          : {metrics['specificity']:.4f}")
    print(f" F1-Score             : {metrics['f1_score']:.4f}")
    print("-" * 65)
    print(" 2x2 Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"   [[TN: {metrics['tn']:<4}, FP: {metrics['fp']:<4}],")
    print(f"    [FN: {metrics['fn']:<4}, TP: {metrics['tp']:<4}]]")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()