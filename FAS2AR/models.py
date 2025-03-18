import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    """
    A simple 1D CNN embedding network.
    Input shape: (batch_size, channels, seq_length)
    Output: embedding vector of fixed dimension.
    """
    def __init__(self, input_channels=3, seq_length=100, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        # Global average pooling over the time dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)
    
    def forward(self, x):
        # x: (batch_size, channels, seq_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)  # shape: (batch_size, 128, 1)
        x = x.squeeze(-1)        # shape: (batch_size, 128)
        embedding = self.fc(x)   # shape: (batch_size, embedding_dim)
        return embedding

class Adapter(nn.Module):
    """
    A small adapter network that is attached on top of the frozen embedding.
    It maps the embedding into class logits.
    """
    def __init__(self, embedding_dim=128, num_classes=5):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, embedding):
        x = self.fc1(embedding)
        x = self.relu(x)
        out = self.fc2(x)
        return out

def extract_style(x, eps=1e-6):
    """
    Computes a style vector from sensor input by computing the channel-wise mean and std.
    
    Args:
        x: Input tensor of shape (batch_size, channels, seq_length).
    
    Returns:
        style: Tensor of shape (batch_size, channels*2) where the first half is the mean 
               and the second half is the std.
    """
    # Compute mean and standard deviation along the time dimension (dim=2)
    mean = x.mean(dim=2)  # shape: (batch_size, channels)
    std = x.std(dim=2)    # shape: (batch_size, channels)
    style = torch.cat([mean, std], dim=1)
    return style

def synthesize_with_style(x, style, eps=1e-6):
    """
    Generates a new sensor signal by re-normalizing x with a given style.
    Here, we adjust each channel of x to have the provided mean and std.
    
    Args:
        x: Original sensor signal tensor of shape (batch_size, channels, seq_length).
        style: Style vector of shape (batch_size, channels*2), where the first half is new mean 
               and the second half is new std.
    
    Returns:
        x_new: Sensor signal re-synthesized with the new style.
    """
    batch_size, channels, seq_length = x.size()
    # Compute original per-channel statistics (over the time dimension)
    orig_mean = x.mean(dim=2, keepdim=True)  # shape: (batch_size, channels, 1)
    orig_std = x.std(dim=2, keepdim=True)    # shape: (batch_size, channels, 1)
    
    # Extract new style parameters
    new_mean = style[:, :channels].unsqueeze(2)  # shape: (batch_size, channels, 1)
    new_std = style[:, channels:].unsqueeze(2)     # shape: (batch_size, channels, 1)
    
    # Normalize x and reapply the new style
    x_norm = (x - orig_mean) / (orig_std + eps)
    x_new = new_std * x_norm + new_mean
    return x_new