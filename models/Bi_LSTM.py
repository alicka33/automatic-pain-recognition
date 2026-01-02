import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # Dodajemy dla Bi-LSTM

        # 1. Warstwa LSTM: bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        # Warstwa Dropout dla wyjścia z LSTM
        self.dropout = nn.Dropout(dropout_prob)

        # 2. Dostosowanie Warstwy Wycjściowej (Klasyfikatora)
        # Wejście musi być podwójne, ponieważ Bi-LSTM zwraca hidden_size * 2
        # Używamy hidden_size * num_directions (czyli hidden_size * 2)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x kształt: (Batch Size, Sequence Length, Features)

        # 1. Przekazanie przez Bi-LSTM
        # out: (Batch Size, Sequence Length, hidden_size * 2)
        out, (h_n, c_n) = self.lstm(x)

        # 2. Agregacja w czasie (Temporal Attention - Uproszczone)
        # Zamiast brać tylko ostatnią klatkę: out = out[:, -1, :]
        # Bierzemy ŚREDNIĄ Wartość ze WSZYSTKICH KROKÓW CZASOWYCH.
        # Pomaga to zapobiec ignorowaniu ważnych, ale nieostatnich klatek.
        # agg_out kształt: (Batch Size, hidden_size * 2)
        agg_out = torch.mean(out, dim=1)

        # 3. Dropout i Warstwa wyjściowa
        agg_out = self.dropout(agg_out)
        final_output = self.fc(agg_out)  # Kształt: (Batch Size, num_classes)

        return final_output
