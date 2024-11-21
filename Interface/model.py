# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes_action=9, num_classes_offence=3, num_classes_severity=5,
                 num_classes_bodypart=2, num_classes_offence_severity=4):
        super(TwoStreamNetwork, self).__init__()

        # Backbones for RGB and Flow
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.flow_backbone = models.resnet50(weights='IMAGENET1K_V1')

        # Replace the final layers
        num_ftrs = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()
        self.flow_backbone.fc = nn.Identity()

        # Fully connected layers
        self.fc_action = nn.Linear(num_ftrs * 2, num_classes_action)
        self.fc_offence = nn.Linear(num_ftrs * 2, num_classes_offence)
        self.fc_severity = nn.Linear(num_ftrs * 2, num_classes_severity)
        self.fc_bodypart = nn.Linear(num_ftrs * 2, num_classes_bodypart)
        self.fc_offence_severity = nn.Linear(num_ftrs * 2, num_classes_offence_severity)

    def forward(self, rgb_input, flow_input):
        # Reshape input tensors for per-frame processing
        batch_size, num_frames, channels, height, width = rgb_input.shape
        rgb_input = rgb_input.view(batch_size * num_frames, channels, height, width)  # [batch_size * num_frames, 3, 112, 112]
        
        # Repeat for flow_input
        flow_input = flow_input.view(batch_size * num_frames, 1, height, width).repeat(1, 3, 1, 1)  # [batch_size * num_frames, 3, 112, 112]

        # Process each frame through the ResNet backbone
        rgb_features = self.rgb_backbone(rgb_input)  # [batch_size * num_frames, num_ftrs]
        flow_features = self.flow_backbone(flow_input)  # [batch_size * num_frames, num_ftrs]

        # Reshape back to original batch and frame dimensions
        rgb_features = rgb_features.view(batch_size, num_frames, -1)
        flow_features = flow_features.view(batch_size, num_frames, -1)

        # Aggregate across frames
        rgb_features = rgb_features.mean(dim=1)  # [batch_size, num_ftrs]
        flow_features = flow_features.mean(dim=1)  # [batch_size, num_ftrs]

        # Combine both features
        combined_features = torch.cat((rgb_features, flow_features), dim=1)  # [batch_size, num_ftrs * 2]

        # Forward through task-specific layers
        action_out = self.fc_action(combined_features)
        offence_out = self.fc_offence(combined_features)
        severity_out = self.fc_severity(combined_features)
        bodypart_out = self.fc_bodypart(combined_features)
        offence_severity_out = self.fc_offence_severity(combined_features)

        return action_out, offence_out, severity_out, bodypart_out, offence_severity_out

def load_model(model_path: str):
    model = TwoStreamNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
    model.eval()
    return model
