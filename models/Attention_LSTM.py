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
                            dropout=dropout_prob,
                            bidirectional=True)

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.layer_norm(out)

        logits = self.attention_weights_layer(out)

        logits = logits.squeeze(2)

        weights = F.softmax(logits, dim=1)

        weighted_output = out * weights.unsqueeze(2)
        agg_out = torch.sum(weighted_output, dim=1)

        agg_out = self.dropout(agg_out)
        final_output = self.fc(agg_out)

        return final_output
