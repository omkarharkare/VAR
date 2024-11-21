import torch
import torch.nn as nn
import torchvision.models as models

class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes_action=9, num_classes_offence=3, num_classes_severity=5,
                 num_classes_bodypart=2, num_classes_offence_severity=4, freeze_backbone=True):
        super(TwoStreamNetwork, self).__init__()

        # Load the backbone for both streams
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.flow_backbone = models.resnet50(weights='IMAGENET1K_V1')

        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False
            for param in self.flow_backbone.parameters():
                param.requires_grad = False

        # Replace the final layer with Identity for both backbones
        num_ftrs = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()  # RGB Stream
        self.flow_backbone.fc = nn.Identity()  # Optical Flow Stream

        # Define fully connected layers for classification
        self.fc_action = nn.Linear(num_ftrs * 2, num_classes_action)
        self.fc_offence = nn.Linear(num_ftrs * 2, num_classes_offence)
        self.fc_severity = nn.Linear(num_ftrs * 2, num_classes_severity)
        self.fc_bodypart = nn.Linear(num_ftrs * 2, num_classes_bodypart)
        self.fc_offence_severity = nn.Linear(num_ftrs * 2, num_classes_offence_severity)

    def forward(self, rgb_input, flow_input):
        # Extract batch size and frame count from the RGB input
        batch_size, num_streams, num_frames, _, _, _ = rgb_input.shape  # Shape: [batch_size, num_streams, num_frames, 3, 112, 112]

        # Reshape input tensors for per-frame processing
        rgb_input = rgb_input.view(batch_size * num_streams * num_frames, 3, 112, 112)  # Flatten to [batch_size * num_streams * num_frames, 3, 56, 56]
        flow_input = flow_input.view(batch_size * num_streams * num_frames, 1, 112, 112)  # Flatten to [batch_size * num_streams * num_frames, 1, 56, 56]
        flow_input = flow_input.repeat(1, 3, 1, 1)  # Repeat the single channel three times

        # Process each frame through backbones
        rgb_features = self.rgb_backbone(rgb_input)  # Shape: [batch_size * num_streams * num_frames, num_ftrs]
        flow_features = self.flow_backbone(flow_input)  # Shape: [batch_size * num_streams * num_frames, num_ftrs]

        # Reshape back to [batch_size, num_streams, num_frames, num_ftrs]
        rgb_features = rgb_features.view(batch_size, num_streams, num_frames, -1)
        flow_features = flow_features.view(batch_size, num_streams, num_frames, -1)

        # Aggregate features across frames (mean pooling over frames)
        rgb_features = rgb_features.mean(dim=2)  # Shape: [batch_size, num_streams, num_ftrs]
        flow_features = flow_features.mean(dim=2)  # Shape: [batch_size, num_streams, num_ftrs]

        # Combine features from both streams
        combined_features = torch.cat((rgb_features, flow_features), dim=2)  # Shape: [batch_size, num_streams, num_ftrs * 2]

        # You may want to aggregate across streams as well, if applicable
        combined_features = combined_features.mean(dim=1)  # Optionally, take mean across streams

        # Forward through task-specific layers
        action_out = self.fc_action(combined_features)
        offence_out = self.fc_offence(combined_features)
        severity_out = self.fc_severity(combined_features)
        bodypart_out = self.fc_bodypart(combined_features)
        offence_severity_out = self.fc_offence_severity(combined_features)

        return action_out, offence_out, severity_out, bodypart_out, offence_severity_out

# # Example of model instantiation
# model = TwoStreamNetwork()

# # Move model to device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Example input for a batch of 2 videos, each with 126 frames
# rgb_input = torch.randn(2, 126, 3, 224, 224).to(device)  # RGB frames
# flow_input = torch.randn(2, 126, 3, 224, 224).to(device)  # Flow frames

# # Forward pass
# outputs = model(rgb_input, flow_input)
# for output in outputs:
#     print(output.shape)  # Should print the shape of each output tensor
