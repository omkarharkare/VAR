##<--- Jainil --->##
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
import tempfile
import cv2
import os
from model import load_model

app = FastAPI()

# Load the model
model = load_model("best_model_1.pth")

EVENT_DICTIONARY = {
    'action_class': {
        0: "Tackling", 1: "Standing tackling", 2: "High leg", 3: "Holding", 
        4: "Pushing", 5: "Elbowing", 6: "Challenge", 7: "Dive", 8: "Don't know"
    },
    'offence_class': {
        0: "Yes", 1: "Between", 2: "No"
    },
    'severity_class': {
        0: "1.0", 1: "2.0", 2: "3.0", 3: "4.0", 4: "5.0"
    },
    'bodypart_class': {
        0: "Upper body", 1: "Under body"
    },
    'offence_severity_class': {
        0: "No offence", 1: "Offence + No card", 
        2: "Yellow card", 3: "Red card"
    }
}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>VAR - Video Assistant Referee</title>
    <style>
        :root {
            --primary-color: #00ff85;
            --secondary-color: #003a70;
            --accent-color: #ff0040;
            --card-bg: rgba(255, 255, 255, 0.95);
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                        url('https://images.unsplash.com/photo-1489944440615-453fc2b6a9a9') center/cover no-repeat fixed;
            color: white;
            overflow: hidden;
        }

        .main-container {
            display: grid;
            grid-template-columns: 250px 1fr;
            height: 100vh;
        }

        .sidebar {
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-right: 1px solid var(--primary-color);
        }

        .content-area {
            display: grid;
            grid-template-rows: 1fr auto;
            padding: 20px;
            gap: 20px;
            overflow: hidden;
        }

        .video-container {
            overflow-y: auto;
            padding-right: 10px;
        }

        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2em;
            margin: 0;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .upload-section {
            margin-top: 20px;
        }

        .custom-file-upload {
            display: block;
            padding: 12px;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: white;
            border-radius: 10px;
            cursor: pointer;
            margin-bottom: 15px;
            text-align: center;
            transition: transform 0.3s ease;
        }

        #uploadBtn {
            width: 100%;
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .video-card {
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
            height: 100%;
        }

        .video-card video {
            width: 100%;
            max-height: 200px;
            object-fit: cover;
        }

        .prediction-section {
            padding: 15px;
            color: #333;
            font-size: 0.9em;
        }

        .aggregated-results {
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid var(--primary-color);
        }

        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .prediction-item {
            margin: 8px 0;
        }

        .confidence-bar {
            background: #e9ecef;
            height: 6px;
            border-radius: 3px;
            margin: 5px 0;
            width: 100%;
        }

        .confidence-value {
            height: 100%;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            transition: width 0.6s ease;
        }

        .loading {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 25px;
            border-radius: 10px;
            display: none;
            align-items: center;
            gap: 10px;
            z-index: 1000;
            border: 1px solid var(--primary-color);
        }

        .loading-spinner {
            border: 3px solid rgba(0, 255, 133, 0.3);
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        #fileInput {
            display: none;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 4px;
        }
    </style>
</head>
<body>
   <div class="main-container">
        <div class="sidebar">
            <div class="header">
                <h1>VAR Analysis</h1>
            </div>
            <div class="upload-section">
                <label class="custom-file-upload">
                    <input type="file" id="fileInput" accept="video/mp4,video/avi,video/mov" multiple>
                    Choose Videos (Max 4)
                </label>
                <button id="uploadBtn" onclick="uploadAndPredict()">Analyze Incident</button>
            </div>
        </div>
        
        <div class="content-area">
            <div class="video-container">
                <div class="video-grid" id="videoGrid"></div>
            </div>
            <div class="aggregated-results" id="aggregatedResults">
                <h3>Final VAR Decision</h3>
                <div class="prediction-grid"></div>
            </div>
        </div>
    </div>

    <div class="loading" id="loading">
        <div class="loading-spinner"></div>
        <span>Analyzing...</span>
    </div>

    <script>
        document.getElementById('fileInput').addEventListener('change', function(e) {
    const videoGrid = document.getElementById('videoGrid');
    videoGrid.innerHTML = '';
    
    Array.from(e.target.files).forEach((file, index) => {
        const card = document.createElement('div');
        card.className = 'video-card';
        
        const video = document.createElement('video');
        video.controls = true;
        video.src = URL.createObjectURL(file);
        
        const predictionSection = document.createElement('div');
        predictionSection.className = 'prediction-section';
        predictionSection.innerHTML = `<h3>Video ${index + 1}: ${file.name}</h3>`;
        
        card.appendChild(video);
        card.appendChild(predictionSection);
        videoGrid.appendChild(card);
    });
});

async function uploadAndPredict() {
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const aggregatedResults = document.getElementById('aggregatedResults');
    
    if (fileInput.files.length === 0) {
        alert('Please select at least one video file');
        return;
    }
    
    if (fileInput.files.length > 4) {
        alert('Maximum 4 videos allowed');
        return;
    }
    
    loading.style.display = 'flex';
    
    const formData = new FormData();
    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('files', fileInput.files[i]);
    }
    
    try {
        const response = await fetch('/api/predict_multiple', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing videos. Please try again.');
    } finally {
        loading.style.display = 'none';
    }
}

function createPredictionItem(label, prediction) {
    return `
        <div class="prediction-item">
            <p>${label}: ${prediction.label}</p>
            <div class="confidence-bar">
                <div class="confidence-value" style="width: ${prediction.confidence * 100}%"></div>
            </div>
            <span>${(prediction.confidence * 100).toFixed(1)}%</span>
        </div>
    `;
}

function displayResults(data) {
    // Update individual video predictions
    const cards = document.querySelectorAll('.video-card');
    
    data.individual_predictions.forEach((pred, index) => {
        const predictionSection = cards[index].querySelector('.prediction-section');
        predictionSection.innerHTML = `
            <h3>Video ${index + 1}: ${pred.video_name}</h3>
            <div class="prediction-grid">
                ${createPredictionItem('Action', pred.action)}
                ${createPredictionItem('Offence', pred.offence)}
                ${createPredictionItem('Severity', pred.severity)}
                ${createPredictionItem('Body Part', pred.bodypart)}
                ${createPredictionItem('Offence Severity', pred.offence_severity)}
            </div>
        `;
    });
    
    // Update aggregated results
    const aggregatedResults = document.getElementById('aggregatedResults');
    const predictionGrid = aggregatedResults.querySelector('.prediction-grid');
    predictionGrid.innerHTML = `
        ${createPredictionItem('Action', data.aggregated_prediction.action)}
        ${createPredictionItem('Offence', data.aggregated_prediction.offence)}
        ${createPredictionItem('Severity', data.aggregated_prediction.severity)}
        ${createPredictionItem('Body Part', data.aggregated_prediction.bodypart)}
        ${createPredictionItem('Offence Severity', data.aggregated_prediction.offence_severity)}
    `;
}
    </script>
</body>
</html>
    """

@app.post("/api/predict_multiple")
async def predict_multiple(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    predictions_list = []
    
    for file in files:
        if not file.filename.endswith((".mp4", ".avi", ".mov")):
            raise HTTPException(status_code=400, detail="Invalid video format. Use MP4, AVI, or MOV.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[-4:]) as tmp_file:
            try:
                content = await file.read()
                tmp_file.write(content)
                video_path = tmp_file.name
                
                # Process video
                rgb_input, flow_input = preprocess_video(video_path)
                
                with torch.no_grad():
                    action_out, offence_out, severity_out, bodypart_out, offence_severity_out = model(rgb_input, flow_input)
                    
                    # Calculate probabilities
                    action_probs = F.softmax(action_out, dim=1)
                    offence_probs = F.softmax(offence_out, dim=1)
                    severity_probs = F.softmax(severity_out, dim=1)
                    bodypart_probs = F.softmax(bodypart_out, dim=1)
                    offence_severity_probs = F.softmax(offence_severity_out, dim=1)
                    
                    prediction = {
                        "video_name": file.filename,
                        "action": {
                            "label": EVENT_DICTIONARY['action_class'][torch.argmax(action_probs, dim=1).item()],
                            "confidence": float(action_probs.max().item())
                        },
                        "offence": {
                            "label": EVENT_DICTIONARY['offence_class'][torch.argmax(offence_probs, dim=1).item()],
                            "confidence": float(offence_probs.max().item())
                        },
                        "severity": {
                            "label": EVENT_DICTIONARY['severity_class'][torch.argmax(severity_probs, dim=1).item()],
                            "confidence": float(severity_probs.max().item())
                        },
                        "bodypart": {
                            "label": EVENT_DICTIONARY['bodypart_class'][torch.argmax(bodypart_probs, dim=1).item()],
                            "confidence": float(bodypart_probs.max().item())
                        },
                        "offence_severity": {
                            "label": EVENT_DICTIONARY['offence_severity_class'][torch.argmax(offence_severity_probs, dim=1).item()],
                            "confidence": float(offence_severity_probs.max().item())
                        }
                    }

                    # Check if offence is 'No' and adjust severity accordingly
                    if EVENT_DICTIONARY['offence_class'][torch.argmax(offence_probs, dim=1).item()] == 'No':
                        prediction["severity"] = {
                            "label": "1.0",
                            "confidence": 1.0
                        }
                    else:
                        prediction["severity"] = {
                            "label": EVENT_DICTIONARY['severity_class'][torch.argmax(severity_probs, dim=1).item()],
                            "confidence": float(severity_probs.max().item())
                        }
                    predictions_list.append(prediction)
            finally:
                # Close any open file handles
                try:
                    if 'rgb_input' in locals():
                        del rgb_input
                    if 'flow_input' in locals():
                        del flow_input
                    import gc
                    gc.collect()
                    # Try to delete the temporary file
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")
    
    # Aggregate predictions
    aggregated_prediction = aggregate_predictions(predictions_list)
    
    return {
        "individual_predictions": predictions_list,
        "aggregated_prediction": aggregated_prediction
    }


def preprocess_video(video_path, num_frames=96, img_size=112):
    cap = cv2.VideoCapture(video_path)
    rgb_frames, flow_frames = [], []
    
    while len(rgb_frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        resized_frame = cv2.resize(frame, (img_size, img_size))
        rgb_frames.append(resized_frame)
        flow_frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY))
    
    cap.release()
    
    rgb_tensor = torch.stack([
        torch.tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)) / 255.0 
        for f in rgb_frames
    ]).unsqueeze(0)
    
    flow_tensor = torch.stack([
        torch.tensor(f).unsqueeze(0) / 255.0 
        for f in flow_frames
    ]).unsqueeze(0)
    
    return rgb_tensor, flow_tensor

def aggregate_predictions(predictions_list):
    if not predictions_list:
        return {}
    
    categories = ['action', 'offence', 'severity', 'bodypart', 'offence_severity']
    aggregated = {}
    
    for category in categories:
        labels = [p[category]['label'] for p in predictions_list]
        confidences = [p[category]['confidence'] for p in predictions_list]
        
        weighted_labels = {}
        for label, conf in zip(labels, confidences):
            weighted_labels[label] = weighted_labels.get(label, 0) + conf
        
        final_label = max(weighted_labels.items(), key=lambda x: x[1])[0]
        avg_confidence = sum(confidences) / len(confidences)
        
        aggregated[category] = {
            "label": final_label,
            "confidence": float(avg_confidence)
        }
    
    return aggregated

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)