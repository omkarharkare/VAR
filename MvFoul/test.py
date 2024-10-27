import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from data_loader import VideoDataset
from model import VideoClassifier  # Ensure this matches your actual model import
from preprocess import prepare_data
import time

def log_with_timestamp(message):
    """Logs a message with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_with_timestamp(f"Using device: {device}")
model = VideoClassifier().to(device)
model.load_state_dict(torch.load('best_model.pth'))  # Load your saved model weights
model.eval()  # Set the model to evaluation mode

# Prepare test data (or validation data)
root_dir = 'SoccerNet/mvfouls'
test_data, test_labels = prepare_data(root_dir, 'test')
test_dataset = VideoDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=48)  # Use appropriate batch size

# Calculate accuracy
all_preds = []
all_labels = []

log_with_timestamp("Starting evaluation...")

with torch.no_grad():
    for batch_idx, (batch_videos, batch_labels) in enumerate(test_loader):
        batch_start_time = time.time()
        log_with_timestamp(f"Processing batch {batch_idx+1}/{len(test_loader)}")

        batch_videos = batch_videos.to(device)
        batch_labels = batch_labels.to(device)
        
        # Process each frame separately and average the results (or use another aggregation strategy)
        outputs_list = []
        for i in range(batch_videos.size(1)):  # Iterate over frames
            frame = batch_videos[:, i]  # Select each frame
            output = model(frame)  # Pass each frame through the model
            outputs_list.append(output)

        # Aggregate outputs (e.g., average over frames)
        outputs_aggregated = torch.mean(torch.stack(outputs_list), dim=0)

        predictions = (torch.sigmoid(outputs_aggregated) > 0.5).float()
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

        log_with_timestamp(f"Batch {batch_idx+1} processed in {time.time() - batch_start_time:.2f}s.")

accuracy = accuracy_score(all_labels, all_preds)
log_with_timestamp(f'Test Accuracy: {accuracy * 100:.2f}%')