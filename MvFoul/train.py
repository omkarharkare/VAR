import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision import transforms
import numpy as np
from model import ImprovedTwoStreamNetwork
from preprocess import load_filtered_clips_and_labels

# Import your model
#from model import TwoStreamNetwork  # Assuming the model code is saved as model.py

# Custom Dataset class
class ActionDataset(Dataset):
    def __init__(self, rgb_clips, flow_clips, labels, transform=None):
        self.rgb_clips = rgb_clips
        self.flow_clips = flow_clips
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rgb_clips)

    def __getitem__(self, idx):
        rgb_frames = self.rgb_clips[idx]
        flow_frames = self.flow_clips[idx]

        # Apply transformation
        if self.transform:
            rgb_frames = [self.transform(frame) if not isinstance(frame, torch.Tensor) else frame for frame in rgb_frames]
            flow_frames = [self.transform(frame) if not isinstance(frame, torch.Tensor) else frame for frame in flow_frames]

        # Ensure dimensions are [num_frames, channels, height, width]
        rgb_frames = torch.stack(rgb_frames, dim=0)
        flow_frames = torch.stack(flow_frames, dim=0)

        label_dict = {key: torch.tensor(self.labels[key][idx]) for key in self.labels.keys()}

        return rgb_frames, flow_frames, label_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = {key: [] for key in ['action', 'offence', 'severity', 'bodypart', 'offence_severity']}
    all_labels = {key: [] for key in all_preds.keys()}

    for rgb_input, flow_input, labels in tqdm(dataloader, desc="Training"):
        # Check input shapes and move to device
        rgb_input, flow_input = rgb_input.to(device), flow_input.to(device)

        # Verify dimensions; if missing batch dim, add it
        if len(rgb_input.shape) == 4:
            rgb_input = rgb_input.unsqueeze(0)  # Add batch dim if missing
        if len(flow_input.shape) == 4:
            flow_input = flow_input.unsqueeze(0)

        labels = {key: val.to(device) for key, val in labels.items()}

        optimizer.zero_grad()

        # Forward pass
        outputs = model(rgb_input, flow_input)

        # Compute losses for each task
        loss = 0.0
        for i, task in enumerate(all_preds.keys()):
            task_loss = criterion(outputs[i], labels[task])
            loss += task_loss
            all_preds[task].extend(outputs[i].argmax(dim=1).cpu().numpy())
            all_labels[task].extend(labels[task].cpu().numpy())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    accuracy = {task: accuracy_score(all_labels[task], all_preds[task]) for task in all_preds.keys()}

    return avg_loss, accuracy

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = {key: [] for key in ['action', 'offence', 'severity', 'bodypart', 'offence_severity']}
    all_labels = {key: [] for key in all_preds.keys()}

    with torch.no_grad():
        for rgb_input, flow_input, labels in tqdm(dataloader, desc="Validation"):
            rgb_input, flow_input = rgb_input.to(device), flow_input.to(device)
            labels = {key: val.to(device) for key, val in labels.items()}

            # Forward pass
            outputs = model(rgb_input, flow_input)

            # Compute losses and predictions for each task
            loss = 0.0
            for i, task in enumerate(all_preds.keys()):
                task_loss = criterion(outputs[i], labels[task])
                loss += task_loss
                all_preds[task].extend(outputs[i].argmax(dim=1).cpu().numpy())
                all_labels[task].extend(labels[task].cpu().numpy())

            running_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = {task: accuracy_score(all_labels[task], all_preds[task]) for task in all_preds.keys()}

    return avg_loss, accuracy

def main(data_path, num_epochs=10, batch_size=2, learning_rate=1e-4, max_samples_o=10, max_samples_no =10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_rgb_clips, train_flow_clips, train_labels_action, train_labels_offence, train_labels_severity, train_labels_bodypart, train_labels_offence_severity = \
        load_filtered_clips_and_labels(data_path, "train", max_samples_o, max_samples_no)
    
    valid_rgb_clips, valid_flow_clips, valid_labels_action, valid_labels_offence, valid_labels_severity, valid_labels_bodypart, valid_labels_offence_severity = \
        load_filtered_clips_and_labels(data_path, "valid", max_samples_o, max_samples_no)

    # Organize labels in a dictionary format
    train_labels = {
        "action": train_labels_action,
        "offence": train_labels_offence,
        "severity": train_labels_severity,
        "bodypart": train_labels_bodypart,
        "offence_severity": train_labels_offence_severity
    }
    valid_labels = {
        "action": valid_labels_action,
        "offence": valid_labels_offence,
        "severity": valid_labels_severity,
        "bodypart": valid_labels_bodypart,
        "offence_severity": valid_labels_offence_severity
    }

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = ActionDataset(train_rgb_clips, train_flow_clips, train_labels, transform=transform)
    valid_dataset = ActionDataset(valid_rgb_clips, valid_flow_clips, valid_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = ImprovedTwoStreamNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracies: {train_accuracy}")

        # Validate
        val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Accuracies: {val_accuracy}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")

        torch.save(model.state_dict(), "final_model.pth")



if __name__ == "__main__":
    # Update this path with your actual data path
    DATA_PATH = 'mvfouls'
    main(data_path=DATA_PATH)
