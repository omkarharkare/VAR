import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import logging
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """Dataset for loading and preprocessing video data."""
    
    def __init__(
        self, 
        video_paths: List[str],
        labels: List[int],
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = False
    ):
        """
        Initialize the video dataset.
        
        Args:
            video_paths: List of paths to video files
            labels: List of corresponding labels
            num_frames: Number of frames to sample from each video
            target_size: Target size for frame resizing
            augment: Whether to apply data augmentation
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.target_size = target_size
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def augment_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a frame."""
        if torch.rand(1) > 0.5:
            frame = torch.flip(frame, [2])  # horizontal flip
            
        if torch.rand(1) > 0.5:
            # Random brightness adjustment
            brightness = 0.4 * torch.rand(1) + 0.8
            frame = torch.clamp(frame * brightness, 0, 1)
            
        return frame

    def load_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess a video file.
        """
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                raise ValueError(f"Video has no frames: {video_path}")
                
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.target_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).float() / 255.0
                    frame = frame.permute(2, 0, 1)  # Convert to CxHxW format
                    
                    if self.augment:
                        frame = self.augment_frame(frame)
                        
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                    frames.append(torch.zeros(3, *self.target_size))
            
            cap.release()
            return torch.stack(frames)
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {str(e)}")
            return torch.zeros(self.num_frames, 3, *self.target_size)
            
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        video = self.load_video(self.video_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return video, label