import torch
import torch.nn as nn
import torch.nn.functional as F


class STA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(STA_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # Bi-LSTM
        self.input_size = input_size

        # Bi-LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        # Layer Normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Temporal Attention: learns which frames are important
        # Takes LSTM output and produces weight per frame
        self.temporal_attn = nn.Linear(hidden_size * self.num_directions, 1)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (B, T, F) where F = input_size
        B, T, F = x.size()

        # 1. Bi-LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, H*2)
        lstm_out = self.layer_norm(lstm_out)

        # 2. Temporal Attention - learn which frames matter
        temporal_logits = self.temporal_attn(lstm_out)  # (B, T, 1)
        temporal_weights = F.softmax(temporal_logits, dim=1)  # (B, T, 1)
        
        # 3. Weighted aggregation across time
        context = torch.sum(lstm_out * temporal_weights, dim=1)  # (B, H*2)

        # 4. Classification
        context = self.dropout(context)
        output = self.fc(context)  # (B, num_classes)

        return output
