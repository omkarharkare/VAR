from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import shutil
import tempfile
import uvicorn
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model
class VideoClassifier(nn.Module):
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

# Load and preprocess functions
def preprocess_video(video_path: str, num_frames: int = 16, img_size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
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
    
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames_tensor = torch.stack(frames, dim=0)
    return frames_tensor.unsqueeze(0)

def load_model(model_path: str, device: torch.device, num_frames: int = 16) -> VideoClassifier:
    model = VideoClassifier(num_frames=num_frames)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model: VideoClassifier, video_tensor: torch.Tensor, device: torch.device) -> float:
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        output = model(video_tensor)
    return output.item()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./best_model.pth"  # Adjust path as necessary
num_frames = 16
try:
    model = load_model(model_path, device, num_frames=num_frames)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """Serve the HTML frontend directly from here."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            .hidden { display: none; }
            #uploadForm { margin: 20px auto; }
            #result { margin-top: 20px; font-size: 1.2em; color: #333; }
            #progress-bar { width: 0%; transition: width 0.4s ease; }
        </style>
    </head>
    <body>
        <h1>Upload a Video for Classification</h1>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="fileInput" name="file" accept="video/*" required>
            <button type="submit">Upload and Predict</button>
        </form>
        
        <!-- Preview Section -->
        <div id="preview-section" class="hidden mb-6">
            <video id="video-preview" controls class="w-full rounded-lg" style="max-height: 300px"></video>
        </div>

        <!-- Progress Section -->
        <div id="progress-section" class="hidden mb-6">
            <div class="w-full bg-gray-200 rounded-full h-2.5">
                <div id="progress-bar" class="bg-blue-600 h-2.5 rounded-full progress-bar" style="width: 0%"></div>
            </div>
            <p id="progress-text" class="text-sm text-gray-600 text-center mt-2">Processing video... 0%</p>
        </div>
        
        <div id="result"></div>

        <script>
            document.getElementById("uploadForm").onsubmit = async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById("fileInput");
                if (!fileInput.files.length) return;

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                // Preview video
                const videoPreview = document.getElementById("video-preview");
                videoPreview.src = URL.createObjectURL(fileInput.files[0]);
                document.getElementById("preview-section").classList.remove("hidden");

                // Show progress
                document.getElementById("progress-section").classList.remove("hidden");
                document.getElementById("progress-bar").style.width = "0%";
                document.getElementById("progress-text").textContent = "Processing video... 0%";

                try {
                    const response = await fetch("/api/predict", {
                        method: "POST",
                        body: formData
                    });

                    let progress = 0;
                    const interval = setInterval(() => {
                        if (progress >= 100) clearInterval(interval);
                        else progress += 10;
                        document.getElementById("progress-bar").style.width = `${progress}%`;
                        document.getElementById("progress-text").textContent = `Processing video... ${progress}%`;
                    }, 300);

                    const result = await response.json();
                    
                    if (result.status === "success") {
                        document.getElementById("result").textContent =
                            `Prediction: ${result.prediction}, Confidence: ${result.confidence}`;
                    } else {
                        document.getElementById("result").textContent = "Error processing video.";
                    }

                    clearInterval(interval);
                    document.getElementById("progress-bar").style.width = "100%";
                    document.getElementById("progress-text").textContent = "Processing complete.";

                } catch (error) {
                    document.getElementById("result").textContent = "Error communicating with server.";
                    console.error("Error:", error);
                }
            };
        </script>
    </body>
    </html>
    """


@app.post("/api/predict")
async def classify_video(file: UploadFile = File(...)):
    """API endpoint to classify an uploaded video."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format. Use MP4, AVI, or MOV.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        video_path = tmp_file.name

    try:
        video_tensor = preprocess_video(video_path, num_frames=num_frames)
        probability = predict(model, video_tensor, device)
        prediction = "Foul" if 1 - probability > 0.5 else "No Foul"
        confidence = round(probability if prediction == "No Foul" else 1 - probability, 4)

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Error processing video.")
    finally:
        Path(video_path).unlink()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
