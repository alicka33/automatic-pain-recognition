import torch
import torch.nn as nn
import torch.nn.functional as fun


class STA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(STA_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # dla Bi-LSTM

        # Bi-LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        # Spatial Attention
        self.spatial_attn = nn.Linear(input_size, input_size)  # α_t dla cech

        # Temporal Attention
        self.temporal_attn = nn.Linear(hidden_size * self.num_directions, 1)  # β_t dla klatek

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.size()

        # --- Spatial Attention ---
        # Obliczamy wagi α_t dla każdej cechy
        # α_t = softmax(W_s * x_t)
        spatial_weights = fun.softmax(self.spatial_attn(x), dim=2)  # (B, T, F)
        x_weighted = x * spatial_weights  # ważone punkty

        # --- Bi-LSTM ---
        lstm_out, _ = self.lstm(x_weighted)  # (B, T, hidden*2)

        # --- Temporal Attention ---
        # Obliczamy wagi β dla każdej klatki
        # β_t = softmax(W_t * h_t)
        beta = fun.softmax(self.temporal_attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(lstm_out * beta, dim=1)  # (B, hidden*2)

        # Dropout i klasyfikacja
        context = self.dropout(context)
        output = self.fc(context)  # (B, num_classes)

        return output
