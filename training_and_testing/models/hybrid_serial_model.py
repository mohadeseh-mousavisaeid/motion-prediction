import torch
import torch.nn as nn
from models.data_based_models import LSTMModel
from models.physics_based_models import SingleTrackModel
from models.physics_based_models import ConstantVelocityModel


class HybridSerialModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_output_dim):
        super(HybridSerialModel, self).__init__()
        self.single_track_model = SingleTrackModel(dt=1.0)
        self.lstm_model = LSTMModel(lstm_input_dim, lstm_hidden_dim, lstm_output_dim)
        
        
        
    def forward(self, x, flag):
        if(flag):
            output = LSTMModel(self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_output_dim)
        else:
            output = ConstantVelocityModel()
            
            
        return output