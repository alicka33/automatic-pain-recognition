import torch
import torch.nn as nn
import torch.nn.functional as F


class STA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(STA_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # Bi-LSTM

        # Bi-LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        # Layer Normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Spatial Attention: learns which features are important
        # Input: (B, T, H*2) → Output: (B, T, 1) for each frame
        self.spatial_attn = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)  # Attention weight per input feature
        )

        # Temporal Attention: learns which frames are important
        # Input: (B, T, H*2) → Output: (B, T, 1) for each frame
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Single weight per frame
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (B, T, F) where F = input_size
        B, T, F = x.size()

        # --- Bi-LSTM ---
        lstm_out, _ = self.lstm(x)  # (B, T, H*2)
        lstm_out = self.layer_norm(lstm_out)

        # --- Spatial Attention (on input features) ---
        # Learn which of the F input features are important
        spatial_logits = self.spatial_attn(lstm_out)  # (B, T, F)
        spatial_weights = F.softmax(spatial_logits, dim=2)  # Softmax over features (dim 2)
        x_attended = x * spatial_weights  # Weight the original input features

        # --- Temporal Attention (on frames) ---
        # Learn which of the T frames are important
        temporal_logits = self.temporal_attn(lstm_out)  # (B, T, 1)
        temporal_weights = F.softmax(temporal_logits, dim=1)  # Softmax over time (dim 1)

        # Aggregate using temporal attention weights
        context = torch.sum(lstm_out * temporal_weights, dim=1)  # (B, H*2)

        # --- Classification ---
        context = self.dropout(context)
        output = self.fc(context)  # (B, num_classes)

        return output
