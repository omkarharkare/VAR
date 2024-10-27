from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
import tempfile
import shutil
from model import VideoClassifier
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_frames=16, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

def preprocess_video(video_path, num_frames=16):
    """Preprocess video file for model input."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
            
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            raise ValueError(f"Video too short. Need at least {num_frames} frames")
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at index {idx}")
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)
        
        cap.release()
        
        frames = torch.FloatTensor(np.array(frames))
        frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.post("/api/classify-video")
async def classify_video(video: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid video format. Please upload MP4, AVI, or MOV file")
    
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        temp_path = temp_dir / f"temp_{video.filename}"
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        try:
            # Preprocess video
            inputs = preprocess_video(temp_path)
            inputs = inputs.to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(inputs)
                probability = outputs.squeeze().item()
                prediction = "Foul" if probability > 0.5 else "No Foul"
            
            return {
                "status": "success",
                "prediction": prediction,
                "confidence": float(probability if prediction == "Foul" else 1 - probability)
            }
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
            
    except Exception as e:
        logger.error(f"Error in file handling: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error handling video: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)