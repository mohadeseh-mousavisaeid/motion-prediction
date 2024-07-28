import torch
import torch.nn as nn
from models.data_based_models import LSTMModel
from models.physics_based_models import SingleTrackModel


class HybridParallelModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_output_dim):
        super(HybridParallelModel, self).__init__()
        self.single_track_model = SingleTrackModel(dt=1.0)
        self.lstm_model = LSTMModel(lstm_input_dim, lstm_hidden_dim, lstm_output_dim)
        
        # Assuming both models output the same dimensionality, otherwise adjust accordingly
        combined_output_dim = lstm_output_dim + 5  # lstm_output_dim from LSTM + 5 from SingleTrackModel
        self.fc_combined = nn.Linear(combined_output_dim, combined_output_dim)

    def forward(self, x):
        single_track_output = self.single_track_model(x)  # shape: (batch_size, 5)
        lstm_output = self.lstm_model(x)  # shape: (batch_size, lstm_output_dim)
        
        combined_output = torch.cat((single_track_output, lstm_output), dim=1)  # shape: (batch_size, combined_output_dim)
        final_output = self.fc_combined(combined_output)  # shape: (batch_size, combined_output_dim)
        
        return final_output