import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from model import VideoClassifier
from data_loader import VideoDataset
from preprocess import prepare_data

def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Calculate class weights inversely proportional to class frequencies."""
    class_counts = torch.bincount(labels.long())
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights

def get_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Create a weighted sampler to balance class distribution."""
    class_counts = torch.bincount(labels.long())
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[labels.long()]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

def evaluate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor) -> Dict[str, float]:
    """Calculate various classification metrics."""
    # Ensure correct shapes
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    y_prob = y_prob.squeeze()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true.cpu(), y_pred.cpu(), average='binary'
    )
    ap = average_precision_score(y_true.cpu(), y_prob.cpu())
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap
    }

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    class_weights: torch.Tensor
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the video classification model with class balance handling."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = Path("experiments") / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        model = VideoClassifier(
            num_frames=config["num_frames"],
            dropout_rate=config["dropout_rate"]
        ).to(device)
        
        # Modified BCE loss without passing weights directly
        criterion = nn.BCELoss()
        
        # Create per-sample weights based on class weights
        def get_sample_weights(labels, class_weights):
            return class_weights[labels.long()].to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
        
        writer = SummaryWriter(exp_dir / "logs")
        
        best_val_f1 = 0
        early_stopping_counter = 0
        history = {
            "train_loss": [], "val_loss": [], 
            "train_metrics": [], "val_metrics": []
        }
        
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0
            train_true = []
            train_pred = []
            train_prob = []
            
            for videos, labels in train_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(videos)
                
                # Get sample weights for this batch
                sample_weights = get_sample_weights(labels, class_weights)
                
                # Calculate weighted loss manually
                loss = criterion(outputs.squeeze(), labels.float())
                weighted_loss = (loss * sample_weights).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                train_loss += weighted_loss.item()
                predictions = (outputs.squeeze() > 0.5).float()
                
                train_true.extend(labels.cpu().numpy())
                train_pred.extend(predictions.cpu().numpy())
                train_prob.extend(outputs.detach().squeeze().cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            train_metrics = evaluate_metrics(
                torch.tensor(train_true),
                torch.tensor(train_pred),
                torch.tensor(train_prob)
            )
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_true = []
            val_pred = []
            val_prob = []
            
            with torch.no_grad():
                for videos, labels in val_loader:
                    videos = videos.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(videos)
                    
                    # Get sample weights for this batch
                    sample_weights = get_sample_weights(labels, class_weights)
                    
                    # Calculate weighted loss manually
                    loss = criterion(outputs.squeeze(), labels.float())
                    weighted_loss = (loss * sample_weights).mean()
                    
                    val_loss += weighted_loss.item()
                    predictions = (outputs.squeeze() > 0.5).float()
                    
                    val_true.extend(labels.cpu().numpy())
                    val_pred.extend(predictions.cpu().numpy())
                    val_prob.extend(outputs.squeeze().cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_metrics = evaluate_metrics(
                torch.tensor(val_true),
                torch.tensor(val_pred),
                torch.tensor(val_prob)
            )
            
            # Update learning rate based on F1 score
            scheduler.step(val_metrics['f1'])
            
            # Log metrics
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            for metric in ['precision', 'recall', 'f1', 'ap']:
                writer.add_scalar(f'Metrics/train_{metric}', train_metrics[metric], epoch)
                writer.add_scalar(f'Metrics/val_{metric}', val_metrics[metric], epoch)
            
            # Update history
            history["train_loss"].append(float(avg_train_loss))  # Convert to float for JSON serialization
            history["val_loss"].append(float(avg_val_loss))
            history["train_metrics"].append({k: float(v) for k, v in train_metrics.items()})
            history["val_metrics"].append({k: float(v) for k, v in val_metrics.items()})
            
            # Save best model based on F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(model.state_dict(), exp_dir / "best_model.pth")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= config["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            print(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}\n"
                f"Train Metrics: Precision: {train_metrics['precision']:.4f}, "
                f"Recall: {train_metrics['recall']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}, "
                f"AP: {train_metrics['ap']:.4f}\n"
                f"Val Metrics: Precision: {val_metrics['precision']:.4f}, "
                f"Recall: {val_metrics['recall']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, "
                f"AP: {val_metrics['ap']:.4f}"
            )
        
        history_path = exp_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        
        model.load_state_dict(torch.load(exp_dir / "best_model.pth", weights_only=True))
        writer.close()
        return model, history
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        writer.close() if 'writer' in locals() else None
        raise

if __name__ == "__main__":
    config = {
        "num_frames": 16,
        "batch_size": 4,
        "epochs": 50,
        "learning_rate": 1e-4,
        "dropout_rate": 0.5,
        "patience": 5
    }
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Prepare data
        root_dir = Path('SoccerNet/mvFouls')
        train_data, train_labels = prepare_data(root_dir, 'train')
        val_data, val_labels = prepare_data(root_dir, 'valid')
        
        # Calculate class weights
        train_labels_tensor = torch.tensor(train_labels)
        class_weights = calculate_class_weights(train_labels_tensor).to(device)
        print(f"Class weights: {class_weights}")
        
        # Create weighted sampler for training data
        sampler = get_weighted_sampler(train_labels_tensor)
        
        # Create datasets
        train_dataset = VideoDataset(
            train_data, 
            train_labels,
            num_frames=config["num_frames"],
            augment=True
        )
        val_dataset = VideoDataset(
            val_data,
            val_labels,
            num_frames=config["num_frames"],
            augment=False
        )
        
        # Create data loaders with weighted sampler for training
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=sampler,  # Use weighted sampler
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Train model
        model, history = train_model(train_loader, val_loader, config, device, class_weights)
        
    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")
        raise