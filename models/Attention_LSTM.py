import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(AttentionSequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob,  # Ten dropout_prob zostanie ZMIENIONY w zmiennych globalnych
                            bidirectional=True)

        # NOWOŚĆ: Normalizacja Warstw (Layer Normalization) dla wyjścia LSTM
        # Kształt: hidden_size * num_directions (128 * 2 = 256)
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # 1. Warstwa Uwagi (Attention Mechanism)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Rzutowanie na wagę (scalar)
        )

        # Dropout na końcu (po agregacji Attention)
        self.dropout = nn.Dropout(dropout_prob)  # Ten dropout_prob zostanie ZMIENIONY

        # 2. Warstwa Wyjściowa (Klasyfikator)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x kształt: (Batch Size, Sequence Length, Features)

        # 1. Przekazanie przez Bi-LSTM
        # out: (B, T, H*2) ; (h_n, c_n): (L*2, B, H)
        out, (h_n, c_n) = self.lstm(x)
        out = self.layer_norm(out)  # out: (B, T, H*2)

        # 2. Obliczenie Wag Uwagi (Attention Weights)
        # 2a. Obliczamy "energię" (nieznormalizowane wagi) dla każdej klatki
        # Kształt logits: (B, T, 1)
        logits = self.attention_weights_layer(out)

        # 2b. Usuwamy wymiar 1 (B, T)
        logits = logits.squeeze(2)

        # 2c. Normalizacja za pomocą Softmax wzdłuż osi czasu (dim=1)
        # weights kształt: (B, T) - suma wag dla każdego przykładu = 1
        weights = F.softmax(logits, dim=1)

        # 3. Ważona Agregacja w Czasie (Weighted Sum)
        # Ważona suma wyjść LSTM: agg_out = sum(out[t] * weights[t])
        # weights kształt: (B, T, 1) -> (B, T, H*2)
        # out kształt: (B, T, H*2)
        # agg_out kształt: (B, H*2)
        weighted_output = out * weights.unsqueeze(2)
        agg_out = torch.sum(weighted_output, dim=1)

        # 4. Dropout i Warstwa wyjściowa
        agg_out = self.dropout(agg_out)
        final_output = self.fc(agg_out)

        return final_output
