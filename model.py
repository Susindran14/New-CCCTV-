import torch
import torch.nn as nn

class PoseStream(nn.Module):
    def __init__(self, dropout=0.5):
        super(PoseStream, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((2, 4, 4)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(32 * 2 * 4 * 4, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.fc(x)

class MotionStream(nn.Module):
    def __init__(self, dropout=0.5):
        super(MotionStream, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((2, 4, 4)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(32 * 2 * 4 * 4, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.fc(x)

class AudioStream(nn.Module):
    def __init__(self, dropout=0.5):
        super(AudioStream, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 5)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(32 * 4 * 5, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.fc(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=128, dropout=0.5):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, pose_feat, motion_feat, audio_feat,
                use_pose=True, use_motion=True, use_audio=True):
        batch_size = pose_feat.size(0)
        device = pose_feat.device

        features = torch.stack([pose_feat, motion_feat, audio_feat], dim=1)

        pose_active = torch.full((batch_size,), bool(use_pose), dtype=torch.bool, device=device)
        motion_active = torch.full((batch_size,), bool(use_motion), dtype=torch.bool, device=device)
        audio_active = torch.full((batch_size,), bool(use_audio), dtype=torch.bool, device=device)

        # Dynamic gating: bypass/zero-out audio if feature tensor is all zeros (e.g., silent audio)
        audio_non_zero = (audio_feat.abs().amax(dim=-1) > 1e-8)
        audio_active = audio_active & audio_non_zero

        active = torch.stack([pose_active, motion_active, audio_active], dim=1)

        if not active.any(dim=1).all():
            fallback = pose_active | motion_active | audio_active
            if not fallback.any():
                raise ValueError("At least one active modality is required for every sample.")
            # For samples with all inactive streams, activate active non-audio modalities
            for i in range(batch_size):
                if not active[i].any():
                    active[i, 0] = bool(use_pose)
                    active[i, 1] = bool(use_motion)

        masked_features = features.masked_fill(~active.unsqueeze(-1), 0.0)

        # PyTorch MultiheadAttention key_padding_mask: True indicates position to mask out
        key_padding_mask = ~active

        attn_output, _ = self.attention(
            masked_features,
            masked_features,
            masked_features,
            key_padding_mask=key_padding_mask
        )

        active_float = active.unsqueeze(-1).to(attn_output.dtype)
        fused_features = (attn_output * active_float).sum(dim=1)
        active_count = active_float.sum(dim=1).clamp_min(1.0)
        fused_features = fused_features / active_count

        fused_features = self.drop(fused_features)
        return self.classifier(fused_features)

class TriStreamViolenceModel(nn.Module):
    def __init__(self, use_pose=True, use_motion=True, use_audio=True, dropout=0.5):
        super(TriStreamViolenceModel, self).__init__()
        self.pose_branch = PoseStream(dropout=dropout)
        self.motion_branch = MotionStream(dropout=dropout)
        self.audio_branch = AudioStream(dropout=dropout)
        self.fusion = CrossAttentionFusion(dropout=dropout)
        self.use_pose = use_pose
        self.use_motion = use_motion
        self.use_audio = use_audio

    def forward(self, pose_in, motion_in, audio_in,
                use_pose=None, use_motion=None, use_audio=None):
        use_pose = self.use_pose if use_pose is None else use_pose
        use_motion = self.use_motion if use_motion is None else use_motion
        use_audio = self.use_audio if use_audio is None else use_audio

        pose_feat = self.pose_branch(pose_in) if use_pose else torch.zeros(pose_in.size(0), 128, device=pose_in.device)
        motion_feat = self.motion_branch(motion_in) if use_motion else torch.zeros(motion_in.size(0), 128, device=motion_in.device)

        is_audio_zero = (audio_in.abs().sum(dim=tuple(range(1, audio_in.ndim))) < 1e-8)
        if not use_audio:
            audio_feat = torch.zeros(audio_in.size(0), 128, device=audio_in.device)
        else:
            audio_feat = self.audio_branch(audio_in)
            if is_audio_zero.any():
                audio_feat = audio_feat.masked_fill(is_audio_zero.unsqueeze(-1), 0.0)

        return self.fusion(
            pose_feat, motion_feat, audio_feat,
            use_pose=use_pose,
            use_motion=use_motion,
            use_audio=use_audio
        )

if __name__ == "__main__":
    print("Initializing Tri-Stream Architecture...")
    model = TriStreamViolenceModel()
    print("Model built successfully! Ready for training.")