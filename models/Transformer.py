import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Dodaje informację o pozycji do cech wejściowych (konieczne dla Transformerów,
    ponieważ Attention jest niezależne od kolejności).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tworzenie macierzy Positional Encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Macierz PE: (max_len, 1, d_model)
        pe = torch.zeros(max_len, 1, d_model)

        # Wzór sin/cos
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) # Rejestracja jako bufor (nie jako parametr do uczenia)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor o kształcie [sequence_length, batch_size, embedding_dim]
               lub [B, T, D] - w tym przypadku transponujemy.
        """
        # Dopasowanie do oczekiwanego kształtu (T, B, D)
        x = x.transpose(0, 1)  # Kształt (T, B, D)

        # Dodanie kodowania pozycyjnego
        # x kształt (T, B, D), pe kształt (max_len, 1, D)
        x = x + self.pe[:x.size(0)] 

        x = self.dropout(x)

        # Powrót do kształtu (B, T, D)
        x = x.transpose(0, 1)
        return x


# =======================================================
# 2. MODEL TRANSFORMERA
# =======================================================
class TransformerSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.3, num_heads=4):
        """
        Args:
            input_size: Rozmiar cech wejściowych (100)
            hidden_size: Wymiar embeddingu Transformera (d_model). Powinien być równy input_size.
            num_layers: Liczba warstw kodera (Transformer Encoder Layers).
            num_classes: Liczba klas wyjściowych (2 dla binarnej).
            dropout_prob: Dropout w całym modelu.
            num_heads: Liczba głowic uwagi.
        """
        super(TransformerSequenceModel, self).__init__()

        # Sprawdzenie warunku: D_MODEL Transformera musi być podzielne przez NUM_HEADS
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) musi być podzielne przez num_heads ({num_heads})")

        self.d_model = hidden_size

        # 1. Warstwa projekcji cech (jeśli input_size != hidden_size)
        # Ponieważ input_size (100) ma być równe d_model Transformera, 
        # ta warstwa jest opcjonalna, ale dobra dla elastyczności.
        self.feature_project = nn.Linear(input_size, self.d_model)

        # 2. Kodowanie Pozycyjne
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_prob)

        # 3. Warstwa Transformera (Encoder Layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=self.d_model * 4,  # Typowa konwencja dla FeedForward
            dropout=dropout_prob,
            batch_first=True  # Ustawiamy, aby Batch był pierwszym wymiarem (B, T, D)
        )

        # 4. Sam Koder (Stacked Layers)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # 5. Globalna Agregacja Cech (Używamy prostej średniej zamiast Attention w tym modelu)
        # Po Transformerze stosujemy Global Average Pooling wzdłuż osi czasu (T)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Zmieni wymiar (B, T, D) na (B, D, 1)

        # 6. Warstwa Wyjściowa (Klasyfikator)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(self.d_model, num_classes) # Wyjście z puli ma wymiar d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x kształt: (Batch Size, Sequence Length, Features) -> (B, T, 100)

        # 1. Projekcja do d_model
        x = self.feature_project(x) 
        # x kształt: (B, T, d_model=128)

        # 2. Dodanie Kodowania Pozycyjnego
        x = self.pos_encoder(x)
        # x kształt: (B, T, d_model)

        # 3. Przekazanie przez Koder Transformera
        # out kształt: (B, T, d_model)
        out = self.transformer_encoder(x)

        # 4. Globalna Agregacja (Pooling)
        # Transponujemy (B, T, D) -> (B, D, T) dla 1D Pooling
        pooled_out = out.transpose(1, 2) 

        # Pooling zmienia (B, D, T) na (B, D, 1)
        pooled_out = self.pooling(pooled_out) 

        # Usuwamy wymiar 1 (B, D)
        agg_out = pooled_out.squeeze(2) 

        # 5. Warstwa Wyjściowa
        agg_out = self.dropout(agg_out)
        final_output = self.fc(agg_out)

        return final_output
