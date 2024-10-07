import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size=32, num_frames=16, target_size=(224, 224), **kwargs):
        super().__init__(**kwargs)  # Call superclass initializer
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.video_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        processed_videos = [self.load_video(path) for path in batch_x]
        return np.array(processed_videos), np.array(batch_y)

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