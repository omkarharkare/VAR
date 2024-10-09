import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import prepare_data
from data_loader import VideoDataset
from model import FoulClassificationModel

# Set your root directory
root_dir = 'D:\soccernet\mvfouls'

# Hyperparameters
batch_size = 1  # Reduced batch size
accumulation_steps = 4  # Number of gradient accumulation steps
num_epochs = 50
learning_rate = 0.001

# Prepare data
train_data, train_labels = prepare_data(root_dir, 'train')
val_data, val_labels = prepare_data(root_dir, 'valid')
test_data, test_labels = prepare_data(root_dir, 'test')

# Create datasets and data loaders
train_dataset = VideoDataset(train_data, train_labels)
val_dataset = VideoDataset(val_data, val_labels)
test_dataset = VideoDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the model and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = FoulClassificationModel().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
    
    for i, (batch_videos, batch_labels) in enumerate(train_loader):
        batch_videos, batch_labels = batch_videos.to(device), batch_labels.to(device)
        
        outputs = model(batch_videos)
        loss = criterion(outputs, batch_labels.unsqueeze(1))
        loss = loss / accumulation_steps  # Normalize the loss
        loss.backward()
        
        train_loss += loss.item() * accumulation_steps  # Accumulate the original loss
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_videos, batch_labels in val_loader:
            batch_videos, batch_labels = batch_videos.to(device), batch_labels.to(device)
            outputs = model(batch_videos)
            val_loss += criterion(outputs, batch_labels.unsqueeze(1)).item()
            predicted = (outputs > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted.squeeze() == batch_labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model and evaluate on test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_videos, batch_labels in test_loader:
        batch_videos, batch_labels = batch_videos.to(device), batch_labels.to(device)
        outputs = model(batch_videos)
        test_loss += criterion(outputs, batch_labels.unsqueeze(1)).item()
        predicted = (outputs > 0.5).float()
        total += batch_labels.size(0)
        correct += (predicted.squeeze() == batch_labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test accuracy: {test_accuracy:.2f}%")

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')