import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from typing import List

class VideoClassifier(nn.Module):
    """3D CNN for video classification."""
    
    def __init__(self, num_frames: int = 16, input_channels: int = 3, dropout_rate: float = 0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((num_frames, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256 * num_frames, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W] -> [B, C, F, H, W]
        x = self.features(x)
        return self.classifier(x)

def preprocess_video(video_path: str, num_frames: int = 16, img_size: int = 224) -> torch.Tensor:
    """Load a video, resize frames, and return a tensor with the shape [1, num_frames, 3, img_size, img_size]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read and process frames
    frame_count = 0
    while frame_count < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = transform(frame)
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    # If we have fewer frames, duplicate the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames_tensor = torch.stack(frames, dim=0)  # Shape: [num_frames, 3, img_size, img_size]
    return frames_tensor.unsqueeze(0)  # Shape: [1, num_frames, 3, img_size, img_size]

def load_model(model_path: str, device: torch.device, num_frames: int = 16) -> VideoClassifier:
    """Load a pretrained model from a .pth file."""
    model = VideoClassifier(num_frames=num_frames)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model: VideoClassifier, video_tensor: torch.Tensor, device: torch.device) -> float:
    """Perform inference and return the probability of the positive class."""
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        output = model(video_tensor)
    return output.item()  # Return the probability of the positive class

if __name__ == "__main__":
    # Paths and parameters
    video_path = "./clip_3.mp4"  # Replace with the path to your video file
    model_path = "./best_model.pth"       # Replace with the path to your model .pth file
    num_frames = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess video
    video_tensor = preprocess_video(video_path, num_frames=num_frames)
    print(f"Processed video tensor shape: {video_tensor.shape}")
    
    # Load model
    model = load_model(model_path, device, num_frames=num_frames)
    print("Model loaded successfully.")
    
    # Make a prediction
    probability = predict(model, video_tensor, device)
    print(f"Predicted probability of positive class or non-foul: {probability:.4f}")
