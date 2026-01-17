import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)

        agg_out = torch.mean(out, dim=1)

        agg_out = self.dropout(agg_out)
        final_output = self.fc(agg_out)

        return final_output
