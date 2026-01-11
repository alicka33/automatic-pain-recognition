import torch
import torch.nn as nn
import torch.nn.functional as F


class STA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # Bi-LSTM

        # Bi-LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True
        )

        # LayerNorm for stability
        self.ln_lstm = nn.LayerNorm(hidden_size * self.num_directions)

        # Spatial attention: which features matter per frame (B, T, F)
        self.spatial_attn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        nn.init.xavier_uniform_(self.spatial_attn[0].weight)
        nn.init.zeros_(self.spatial_attn[0].bias)
        nn.init.xavier_uniform_(self.spatial_attn[2].weight)
        nn.init.zeros_(self.spatial_attn[2].bias)

        # Temporal attention: which frames matter (B, T, 1)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        nn.init.xavier_uniform_(self.temporal_attn[0].weight)
        nn.init.zeros_(self.temporal_attn[0].bias)
        nn.init.xavier_uniform_(self.temporal_attn[2].weight)
        nn.init.zeros_(self.temporal_attn[2].bias)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F_dim = x.size()

        # Spatial attention on inputs
        spatial_logits = self.spatial_attn(x)              # (B, T, F)
        spatial_weights = F.softmax(spatial_logits, dim=2) # softmax over features
        x_weighted = x * spatial_weights                   # reweighted features

        # Bi-LSTM
        lstm_out, _ = self.lstm(x_weighted)                # (B, T, H*2)
        lstm_out = self.ln_lstm(lstm_out)

        # Temporal attention with padding mask (assumes padded frames are all zeros)
        temporal_logits = self.temporal_attn(lstm_out)     # (B, T, 1)
        frame_mask = (x.abs().sum(dim=2) == 0)             # (B, T)
        temporal_logits = temporal_logits.masked_fill(frame_mask.unsqueeze(2), -1e9)
        temporal_weights = F.softmax(temporal_logits, dim=1)  # softmax over time

        context = torch.sum(lstm_out * temporal_weights, dim=1)  # (B, H*2)
        context = self.dropout(context)
        return self.fc(context)                             # (B, num_classes)