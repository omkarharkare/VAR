# main.py
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
model = load_model("2_stream_best_model.pth")

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
        2: "Offence + Yellow card", 3: "Offence + Red card"
    }
}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Foul Classification</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-section { margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 5px; }
            .video-preview { margin: 20px 0; max-width: 100%; }
            #results { margin-top: 20px; text-align: left; }
            .loading { display: none; margin: 20px 0; }
            .prediction { margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Foul Classification</h1>
            
            <div class="upload-section">
                <h2>Choose Videos</h2>
                <input type="file" id="videoInput" name="files" accept="video/mp4,video/avi,video/mov" multiple>
                <button onclick="uploadAndPredict()">Upload and Predict</button>
            </div>

            <div id="loading" class="loading">
                Processing videos... Please wait.
            </div>

            <div id="results"></div>
        </div>

        <script>
            async function uploadAndPredict() {
                const fileInput = document.getElementById('videoInput');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');

                if (fileInput.files.length === 0) {
                    alert('Please select at least one video file');
                    return;
                }

                if (fileInput.files.length > 4) {
                    alert('Maximum 4 videos allowed');
                    return;
                }

                loading.style.display = 'block';
                results.innerHTML = '';

                const formData = new FormData();
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append('files', fileInput.files[i]);
                }

                try {
                    const response = await fetch('/api/predict_multiple', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Upload failed');
                    }

                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    console.error('Error:', error);
                    results.innerHTML = '<p style="color: red;">Error processing videos. Please try again.</p>';
                } finally {
                    loading.style.display = 'none';
                }
            }

            function displayResults(data) {
                const results = document.getElementById('results');
                let html = '';

                html += '<h2>Individual Predictions</h2>';
                data.individual_predictions.forEach((pred, index) => {
                    html += `
                        <div class="prediction">
                            <h3>Video ${index + 1}: ${pred.video_name}</h3>
                            <p>Action: ${pred.action.label} (${(pred.action.confidence * 100).toFixed(2)}%)</p>
                            <p>Offence: ${pred.offence.label} (${(pred.offence.confidence * 100).toFixed(2)}%)</p>
                            <p>Severity: ${pred.severity.label} (${(pred.severity.confidence * 100).toFixed(2)}%)</p>
                            <p>Body Part: ${pred.bodypart.label} (${(pred.bodypart.confidence * 100).toFixed(2)}%)</p>
                            <p>Offence Severity: ${pred.offence_severity.label} (${(pred.offence_severity.confidence * 100).toFixed(2)}%)</p>
                        </div>
                    `;
                });

                html += `
                    <h2>Aggregated Prediction</h2>
                    <div class="prediction">
                        <p>Action: ${data.aggregated_prediction.action.label} (${(data.aggregated_prediction.action.confidence * 100).toFixed(2)}%)</p>
                        <p>Offence: ${data.aggregated_prediction.offence.label} (${(data.aggregated_prediction.offence.confidence * 100).toFixed(2)}%)</p>
                        <p>Severity: ${data.aggregated_prediction.severity.label} (${(data.aggregated_prediction.severity.confidence * 100).toFixed(2)}%)</p>
                        <p>Body Part: ${data.aggregated_prediction.bodypart.label} (${(data.aggregated_prediction.bodypart.confidence * 100).toFixed(2)}%)</p>
                        <p>Offence Severity: ${data.aggregated_prediction.offence_severity.label} (${(data.aggregated_prediction.offence_severity.confidence * 100).toFixed(2)}%)</p>
                    </div>
                `;

                results.innerHTML = html;
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