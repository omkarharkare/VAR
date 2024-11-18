# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from model import TwoStreamNetwork, load_model
import torch
import torch.nn.functional as F
import tempfile
import cv2

app = FastAPI()

# Load the model
model = load_model("best_model_1.pth")

# Load the EVENT_DICTIONARY for mapping annotation labels
EVENT_DICTIONARY = {
    'action_class': {
        0: "Tackling", 1: "Standing tackling", 2: "High leg", 3: "Holding", 4: "Pushing",
        5: "Elbowing", 6: "Challenge", 7: "Dive", 8: "Don't know"
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
        0: "No offence", 1: "Offence + No card", 2: "Offence + Yellow card", 3: "Offence + Red card"
    }
}

@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("index.html").read_text()

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format. Use MP4, AVI, or MOV.")
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.file.read())
        video_path = tmp_file.name

    # Preprocess the video
    rgb_input, flow_input = preprocess_video(video_path)

    # Perform prediction and calculate confidence
    with torch.no_grad():
        action_out, offence_out, severity_out, bodypart_out, offence_severity_out = model(rgb_input, flow_input)
        
        # Calculate confidence for each output by applying softmax
        action_probs = F.softmax(action_out, dim=1)
        offence_probs = F.softmax(offence_out, dim=1)
        severity_probs = F.softmax(severity_out, dim=1)
        bodypart_probs = F.softmax(bodypart_out, dim=1)
        offence_severity_probs = F.softmax(offence_severity_out, dim=1)

        # Get the predicted classes and map to labels
        action_index = torch.argmax(action_probs, dim=1).item()
        offence_index = torch.argmax(offence_probs, dim=1).item()
        severity_index = torch.argmax(severity_probs, dim=1).item()
        bodypart_index = torch.argmax(bodypart_probs, dim=1).item()
        offence_severity_index = torch.argmax(offence_severity_probs, dim=1).item()

        # Map indices to labels using EVENT_DICTIONARY
        action_label = EVENT_DICTIONARY['action_class'][action_index]
        offence_label = EVENT_DICTIONARY['offence_class'][offence_index]
        severity_label = EVENT_DICTIONARY['severity_class'][severity_index]
        bodypart_label = EVENT_DICTIONARY['bodypart_class'][bodypart_index]
        offence_severity_label = EVENT_DICTIONARY['offence_severity_class'][offence_severity_index]

        # Get confidence scores
        action_confidence = action_probs[0, action_index].item()
        offence_confidence = offence_probs[0, offence_index].item()
        severity_confidence = severity_probs[0, severity_index].item()
        bodypart_confidence = bodypart_probs[0, bodypart_index].item()
        offence_severity_confidence = offence_severity_probs[0, offence_severity_index].item()

    # Clean up the temporary video file
    Path(video_path).unlink()
    
    # Return the labels and confidence scores in response
    return {
        "action": {"label": action_label, "confidence": action_confidence},
        "offence": {"label": offence_label, "confidence": offence_confidence},
        "severity": {"label": severity_label, "confidence": severity_confidence},
        "bodypart": {"label": bodypart_label, "confidence": bodypart_confidence},
        "offence_severity": {"label": offence_severity_label, "confidence": offence_severity_confidence}
    }

def preprocess_video(video_path, num_frames=16, img_size=112):
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
    rgb_tensor = torch.stack([torch.tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)) / 255.0 for f in rgb_frames]).unsqueeze(0)
    flow_tensor = torch.stack([torch.tensor(f).unsqueeze(0) / 255.0 for f in flow_frames]).unsqueeze(0)
    return rgb_tensor, flow_tensor

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)