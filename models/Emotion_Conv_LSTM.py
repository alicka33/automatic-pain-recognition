import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(ConvSequenceModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.cnn_out_features = 32 * (input_size // 2)

        self.lstm = nn.LSTM(input_size=self.cnn_out_features, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape

        x_cnn = x.view(batch_size * seq_len, 1, n_features)

        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = self.pool1(x_cnn)

        x_lstm = x_cnn.view(batch_size, seq_len, -1)

        out, _ = self.lstm(x_lstm)

        agg_out = torch.mean(out, dim=1)

        agg_out = self.dropout(agg_out)
        return self.fc(agg_out)
