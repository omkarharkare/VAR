import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ImprovedTwoStreamNetwork(nn.Module):
    def __init__(self, num_classes_action=9, num_classes_offence=3, num_classes_severity=5,
                 num_classes_bodypart=2, num_classes_offence_severity=4, freeze_backbone=True):
        super(ImprovedTwoStreamNetwork, self).__init__()

        # Load more advanced backbones
        # Using RegNet-Y as it shows better performance than ResNet
        self.rgb_backbone = models.regnet_y_32gf(weights='IMAGENET1K_V2')
        self.flow_backbone = models.regnet_y_32gf(weights='IMAGENET1K_V2')

        if freeze_backbone:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False
            for param in self.flow_backbone.parameters():
                param.requires_grad = False

        num_ftrs = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()
        self.flow_backbone.fc = nn.Identity()

        # Temporal attention mechanism
        encoder_layers = TransformerEncoderLayer(
            d_model=num_ftrs,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = TransformerEncoder(encoder_layers, num_layers=2)

        # Stream fusion module
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_ftrs * 2, num_ftrs),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_ftrs)
        )

        # Task-specific heads with deeper architecture
        self.fc_action = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_action)
        )
        
        self.fc_offence = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_offence)
        )
        
        self.fc_severity = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_severity)
        )
        
        self.fc_bodypart = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_bodypart)
        )
        
        self.fc_offence_severity = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_offence_severity)
        )

        # Learnable temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1))

    def temporal_attention(self, features):
        # Apply scaled dot-product attention
        attention_weights = torch.matmul(features, features.transpose(-2, -1)) / self.temperature
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attended_features = torch.matmul(attention_weights, features)
        return attended_features

    def forward(self, rgb_input, flow_input):
        batch_size, num_streams, num_frames, _, _, _ = rgb_input.shape

        # Reshape inputs
        rgb_input = rgb_input.view(batch_size * num_streams * num_frames, 3, 112, 112)
        flow_input = flow_input.view(batch_size * num_streams * num_frames, 1, 112, 112)
        flow_input = flow_input.repeat(1, 3, 1, 1)

        # Extract features
        rgb_features = self.rgb_backbone(rgb_input)
        flow_features = self.flow_backbone(flow_input)

        # Reshape features
        rgb_features = rgb_features.view(batch_size * num_streams, num_frames, -1)
        flow_features = flow_features.view(batch_size * num_streams, num_frames, -1)

        # Apply temporal attention and transformer encoding
        rgb_features = self.temporal_encoder(rgb_features)
        flow_features = self.temporal_encoder(flow_features)

        # Apply temporal attention
        rgb_features = self.temporal_attention(rgb_features)
        flow_features = self.temporal_attention(flow_features)

        # Global average pooling over frames
        rgb_features = rgb_features.mean(dim=1)
        flow_features = flow_features.mean(dim=1)

        # Reshape to [batch_size, num_streams, features]
        rgb_features = rgb_features.view(batch_size, num_streams, -1)
        flow_features = flow_features.view(batch_size, num_streams, -1)

        # Concatenate and fuse streams
        combined_features = torch.cat((rgb_features, flow_features), dim=-1)
        combined_features = self.fusion_layer(combined_features)
        
        # Average across streams
        combined_features = combined_features.mean(dim=1)

        # Forward through task-specific layers
        action_out = self.fc_action(combined_features)
        offence_out = self.fc_offence(combined_features)
        severity_out = self.fc_severity(combined_features)
        bodypart_out = self.fc_bodypart(combined_features)
        offence_severity_out = self.fc_offence_severity(combined_features)

        # Task-specific predictions
        return action_out, offence_out, severity_out, bodypart_out, offence_severity_out

# def get_loss_weights(dataset):
#     """
#     Calculate class weights for weighted loss functions
#     to handle class imbalance
#     """
#     class_counts = torch.zeros(len(EVENT_DICTIONARY['action_class']))
#     # Calculate class frequencies from dataset
#     # ... (implement based on your dataset)
#     weights = 1.0 / class_counts
#     weights = weights / weights.sum()
#     return weights