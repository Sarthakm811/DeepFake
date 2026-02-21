import torch
import torch.nn as nn
from torchvision.models import xception
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

class VideoModel(nn.Module):
    def __init__(self, seq_len=32):
        super().__init__()
        backbone = xception(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.lstm = nn.LSTM(2048, 512, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(1024, 2)
    
    def forward(self, x):  # [B, T, C, H, W]
        B, T = x.shape[:2]
        features = []
        for t in range(T):
            feat = self.backbone(x[:, t]).squeeze(-1).squeeze(-1)
            features.append(feat)
        seq_feat = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(seq_feat)
        return self.classifier(lstm_out[:, -1])

class VideoDetector:
    def __init__(self, model_path="models/video_xception_bilstm.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VideoModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.SEQ_LEN = 32
    
    def extract_frames(self, video_path, max_frames=100):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / 8))  # 8 FPS
        
        frame_idx = 0
        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            frame_idx += 1
        cap.release()
        return frames
    
    def predict(self, video_path):
        """Predict fake probability"""
        frames = self.extract_frames(video_path)
        
        # Pad/truncate to SEQ_LEN
        frame_tensors = []
        for frame in frames[:self.SEQ_LEN]:
            frame_tensors.append(self.transform(frame))
        while len(frame_tensors) < self.SEQ_LEN:
            frame_tensors.append(frame_tensors[-1])
        
        input_tensor = torch.stack(frame_tensors).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()
        return prob, len(frames)
