import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, target_size=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = self.load_video(self.video_paths[idx])
        label = self.labels[idx]
        return torch.from_numpy(video).permute(3, 0, 1, 2).float(), torch.tensor(label).float()

    def load_video(self, video_path):
        try: 
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // self.num_frames)
            
            for i in range(self.num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.target_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                else:
                    frames.append(np.zeros((*self.target_size, 3), dtype=np.float32))
            
            cap.release()
            
            if len(frames) < self.num_frames:
                frames += [np.zeros((*self.target_size, 3), dtype=np.float32)] * (self.num_frames - len(frames))
            elif len(frames) > self.num_frames:
                frames = frames[:self.num_frames]
            
            return np.array(frames)
        
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            return np.zeros((self.num_frames, *self.target_size, 3), dtype=np.float32)