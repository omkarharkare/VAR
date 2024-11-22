import os
import torch
import json
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the desired frame count
DESIRED_FRAME_COUNT = 126
num_views = 2 # Number of views in the action

# Load the EVENT_DICTIONARY for mapping annotation labels
EVENT_DICTIONARY = {
    'action_class': {"Tackling": 0, "Standing tackling": 1, "High leg": 2, "Holding": 3, "Pushing": 4,
                     "Elbowing": 5, "Challenge": 6, "Dive": 7, "Dont know": 8},
    'offence_class': {"Offence": 0, "Between": 1, "No Offence": 2, "No offence": 2},
    'severity_class': {"1.0": 0, "2.0": 1, "3.0": 2, "4.0": 3, "5.0": 4},
    'bodypart_class': {"Upper body": 0, "Under body": 1},
    'offence_severity_class': {"No offence": 0, "Offence + No card": 1, "Offence + Yellow card": 2, "Offence + Red card": 3}
}

# Transformation for RGB preprocessing
rgb_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformation for flow preprocessing
flow_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def load_filtered_clips_and_labels(DATA_PATH, split, max_samples_o, max_samples_no):
    rgb_clips, flow_clips = [], []
    labels_action, labels_offence, labels_severity, labels_bodypart, labels_offence_severity = [], [], [], [], []

    annotations_path = os.path.join(DATA_PATH, split, "annotations.json")
    print(f"Loading annotations from: {annotations_path}")

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    print(f"Total actions found in annotations: {len(annotations['Actions'])}")

    offence_count, no_offence_count, skipped_actions = 0, 0, 0

    for action_index, (action_key, action_data) in enumerate(annotations['Actions'].items()):
        offence_class = action_data['Offence']
        if (offence_class == "Offence" and offence_count >= max_samples_o) or \
           (offence_class in ["No offence", "No Offence"] and no_offence_count >= max_samples_no):
            continue

        # Map labels to indices using the dictionary
        action_label = EVENT_DICTIONARY['action_class'].get(action_data['Action class'])
        offence_label = EVENT_DICTIONARY['offence_class'].get(offence_class)
        severity_label = EVENT_DICTIONARY['severity_class'].get(action_data.get('Severity', '1.0'))
        bodypart_label = EVENT_DICTIONARY['bodypart_class'].get(action_data.get('Bodypart', 'Upper body'))
        offence_severity = f"{offence_class} + {EVENT_DICTIONARY['severity_class'].get(severity_label, 'No card')}"
        offence_severity_label = EVENT_DICTIONARY['offence_severity_class'].get(offence_severity, 0)

        # Skip if any label is missing
        if None in [action_label, offence_label, severity_label, bodypart_label, offence_severity_label]:
            skipped_actions += 1
            continue

        action_folder = os.path.join(DATA_PATH, split, f"action_{action_key}")
        if not os.path.exists(action_folder):
            skipped_actions += 1
            continue

        rgb_action_clips, flow_action_clips = [], []
        for clip_idx in range(num_views):
            clip_path = os.path.join(action_folder, f"clip_{clip_idx}.mp4")
            if not os.path.exists(clip_path):
                continue

            cap = cv2.VideoCapture(clip_path)
            ret, prev_frame = cap.read()
            if not ret:
                continue

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            rgb_frames, flow_frames = [], []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process RGB frame
                rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rgb_frame = rgb_transform(rgb_frame).to(device)
                rgb_frames.append(rgb_frame)

                # Process Optical Flow
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = np.clip(flow, -20, 20)  # Clipping to limit extreme values
                flow = ((flow + 20) * (255.0 / 40)).astype(np.uint8)  # Normalizing to 0-255 range
                flow_frame = Image.fromarray(flow[..., 0])  # Taking the horizontal component for simplicity
                flow_frame = flow_transform(flow_frame).to(device)
                flow_frames.append(flow_frame)
                prev_gray = curr_gray

            cap.release()

            # Adjust frame count
            if len(rgb_frames) > DESIRED_FRAME_COUNT:
                indices = np.linspace(0, len(rgb_frames) - 1, DESIRED_FRAME_COUNT).astype(int)
                rgb_frames = [rgb_frames[i] for i in indices]
                flow_frames = [flow_frames[i] for i in indices]
            elif len(rgb_frames) < DESIRED_FRAME_COUNT:
                rgb_frames += [rgb_frames[-1]] * (DESIRED_FRAME_COUNT - len(rgb_frames))
                flow_frames += [flow_frames[-1]] * (DESIRED_FRAME_COUNT - len(flow_frames))

            rgb_action_clips.append(torch.stack(rgb_frames, dim=0))
            flow_action_clips.append(torch.stack(flow_frames, dim=0))

        if rgb_action_clips and flow_action_clips:
            rgb_clips.append(rgb_action_clips)
            flow_clips.append(flow_action_clips)
            labels_action.append(action_label)
            labels_offence.append(offence_label)
            labels_severity.append(severity_label)
            labels_bodypart.append(bodypart_label)
            labels_offence_severity.append(offence_severity_label)

            if offence_class == "Offence":
                offence_count += 1
            else:
                no_offence_count += 1

        if offence_count >= max_samples_o and no_offence_count >= max_samples_no:
            break

    print("\nSummary:")
    print(f"Total actions loaded: {len(rgb_clips)}")
    print(f"Total actions skipped: {skipped_actions}")
    return rgb_clips, flow_clips, labels_action, labels_offence, labels_severity, labels_bodypart, labels_offence_severity
