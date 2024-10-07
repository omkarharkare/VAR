import os
import json
import cv2
import numpy as np

def prepare_data(root_dir, split):
    data = []
    labels = []
    
    split_dir = os.path.join(root_dir, split)
    annotations_path = os.path.join(split_dir, 'annotations.json')
    
    print(f"Processing {split} split:")
    print(f"  Split directory: {split_dir}")
    print(f"  Annotations path: {annotations_path}")
    
    if not os.path.exists(annotations_path):
        print(f"  Error: Annotations file not found")
        return data, labels
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"  Loaded annotations: {type(annotations)}")
    print(f"  Number of actions: {annotations['Number of actions']}")
    
    actions = annotations['Actions']
    for action_id, action_data in actions.items():
        print(f"  Processing action {action_id}")
        
        # Check if the action is an offence
        is_foul = action_data['Offence'] == 'Offence'
        
        # Construct the path to the first clip
        clip_path = os.path.join(split_dir, f"action_{action_id}", "clip_0.mp4")
        
        if os.path.exists(clip_path):
            data.append(clip_path)
            labels.append(1 if is_foul else 0)
        else:
            print(f"  Warning: Clip not found for action {action_id}: {clip_path}")
    
    print(f"  Processed {len(data)} samples")
    print()
    return data, labels

# Usage
root_dir = 'SoccerNet/mvFouls'  # Update this to your actual root directory
train_data, train_labels = prepare_data(root_dir, 'train')
val_data, val_labels = prepare_data(root_dir, 'valid')
test_data, test_labels = prepare_data(root_dir, 'test')

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Print class distribution
from collections import Counter
print("\nClass distribution:")
print(f"Train: {Counter(train_labels)}")
print(f"Validation: {Counter(val_labels)}")
print(f"Test: {Counter(test_labels)}")

# Print a few sample paths to verify
print("\nSample file paths:")
for i in range(min(5, len(train_data))):
    print(train_data[i])