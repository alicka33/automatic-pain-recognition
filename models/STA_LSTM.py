import torch
import torch.nn as nn
import torch.nn.functional as F


class STA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(STA_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob, bidirectional=True)

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        self.temporal_attn = nn.Linear(hidden_size * self.num_directions, 1)
        nn.init.xavier_uniform_(self.temporal_attn.weight)
        nn.init.zeros_(self.temporal_attn.bias)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        B, T, feat_dim = x.size()

        lstm_out, _ = self.lstm(x)              # (B, T, H*2)
        lstm_out = self.layer_norm(lstm_out)

        # Compute frame-wise temporal logits
        temporal_logits = self.temporal_attn(lstm_out)  # (B, T, 1)

        # Build mask for padded frames (rows with all zeros)
        frame_mask = (x.abs().sum(dim=2) == 0)          # (B, T) True where padded
        temporal_logits = temporal_logits.masked_fill(frame_mask.unsqueeze(2), -1e9)

        temporal_weights = torch.softmax(temporal_logits, dim=1)  # (B, T, 1)

        context = torch.sum(lstm_out * temporal_weights, dim=1)   # (B, H*2)
        context = self.dropout(context)
        return self.fc(context)
