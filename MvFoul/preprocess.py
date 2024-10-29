import os
import json

def prepare_data(root_dir, split):
    data = []
    labels = []
    split_dir = os.path.join(root_dir, split)
    annotations_path = os.path.join(split_dir, 'annotations.json')

    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found")
        return data, labels

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    actions = annotations['Actions']

    for action_id, action_data in actions.items():
        is_foul = action_data['Offence'] == 'Offence'
        
        clip_path = os.path.join(split_dir, f"action_{action_id}", "clip_0.mp4")
        
        if os.path.exists(clip_path):
            data.append(clip_path)
            labels.append(1 if is_foul else 0)

    return data, labels

if __name__ == "__main__":
    root_dir = 'mvfouls'
    train_data, train_labels = prepare_data(root_dir, 'train')