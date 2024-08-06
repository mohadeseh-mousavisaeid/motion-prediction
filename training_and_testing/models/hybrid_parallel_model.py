import torch
import torch.nn as nn
from models.data_based_models import LSTMModel
from models.data_based_models import MultiLayerPerceptron
from models.physics_based_models import SingleTrackModel


class HybridParallelModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HybridParallelModel, self).__init__()
        # self.single_track_model = SingleTrackModel()
        self.mlp = MultiLayerPerceptron()
        self.lstm_model = LSTMModel(input_dim, hidden_dim, output_dim)
        
        # Assuming both models output the same dimensionality, otherwise adjust accordingly
        # combined_output_dim = output_dim + 5  # lstm_output_dim from LSTM + 5 from SingleTrackModel
        combined_output_dim = output_dim * 2 # output_dim from LSTM + MLP
        self.fc_combined = nn.Linear(combined_output_dim, combined_output_dim)

    def forward(self, x):
        # single_track_output = self.mlp(x)  # shape: (batch_size, 5)
        mlp_output = self.mlp(x)  # shape: (batch_size, output_dim)
        lstm_output = self.lstm_model(x)  # shape: (batch_size, output_dim)
        
        combined_output = torch.cat((mlp_output, lstm_output), dim=1)  # shape: (batch_size, combined_output_dim)
        final_output = self.fc_combined(combined_output)  # shape: (batch_size, combined_output_dim)
        
        return final_output