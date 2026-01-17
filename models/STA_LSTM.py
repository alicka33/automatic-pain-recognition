import torch
import torch.nn as nn
import torch.nn.functional as F


class STA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True
        )

        self.ln_lstm = nn.LayerNorm(hidden_size * self.num_directions)

        self.spatial_attn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        nn.init.xavier_uniform_(self.spatial_attn[0].weight)
        nn.init.zeros_(self.spatial_attn[0].bias)
        nn.init.xavier_uniform_(self.spatial_attn[2].weight)
        nn.init.zeros_(self.spatial_attn[2].bias)

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
        B, T, F_dim = x.size()

        spatial_logits = self.spatial_attn(x)                
        spatial_weights = torch.sigmoid(spatial_logits)

        x_weighted = x * (1 + spatial_weights)

        lstm_out, _ = self.lstm(x_weighted)                
        lstm_out = self.ln_lstm(lstm_out)

        temporal_logits = self.temporal_attn(lstm_out)      

        temporal_weights = F.softmax(temporal_logits, dim=1)

        context = torch.sum(lstm_out * temporal_weights, dim=1)
        context = self.dropout(context)
        return self.fc(context)
