import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support
from preprocess import prepare_data
from data_loader import VideoDataset
from model import OffsideDetectionModel  # Import the new model
import time

def log_with_timestamp(message):
    """Logs a message with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Set your root directory
root_dir = 'D:/soccernet/mvfouls'

# Hyperparameters
batch_size = 48
num_epochs = 50
learning_rate = 0.01

# Prepare data
log_with_timestamp("Preparing data...")
train_data, train_labels = prepare_data(root_dir, 'train')
val_data, val_labels = prepare_data(root_dir, 'valid')
test_data, test_labels = prepare_data(root_dir, 'test')

# Create datasets and data loaders
log_with_timestamp("Creating datasets and dataloaders...")
train_dataset = VideoDataset(train_data, train_labels)
val_dataset = VideoDataset(val_data, val_labels)
test_dataset = VideoDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the model and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = OffsideDetectionModel().to(device)

# Calculate class weights
pos_weight = torch.tensor([(len(train_labels) - sum(train_labels)) / sum(train_labels)]).to(device)

# Define loss function and optimizer for binary classification
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=learning_rate)  # Only train the last layer

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    log_with_timestamp(f"Epoch {epoch+1}/{num_epochs} started.")
    epoch_start_time = time.time()
    model.train()
    train_loss = 0

    for batch_idx, (batch_videos, batch_labels) in enumerate(train_loader):
        batch_processing_start_time = time.time()
        log_with_timestamp(f"Processing batch {batch_idx+1}/{len(train_loader)}")
        
        batch_videos, batch_labels = batch_videos.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        
        # Process each frame separately and average the results (or use another aggregation strategy)
        outputs_list = []
        for i in range(batch_videos.size(1)):  # Iterate over frames
            outputs = model(batch_videos[:, i])  # Pass each frame through the model
            outputs_list.append(outputs)

        # Aggregate outputs (e.g., average over frames)
        outputs_aggregated = torch.mean(torch.stack(outputs_list), dim=0)

        loss = criterion(outputs_aggregated, batch_labels.unsqueeze(1).float())
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        log_with_timestamp(f"Batch {batch_idx+1} processed in {time.time() - batch_processing_start_time:.2f}s.")

    avg_train_loss = train_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time

    log_with_timestamp(f'Epoch [{epoch+1}/{num_epochs}] completed. Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}')
    torch.save(model.state_dict(), 'final_model.pth')
    

    # Validation step omitted for brevity...

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')
log_with_timestamp("Training completed and model saved.")