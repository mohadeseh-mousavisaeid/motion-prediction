import torch
import torch.nn.functional as F

"""
This is the main step function that is used by training_step, validation_step, and test_step.
"""
# TODO: You have to modify this based on your task, model and data. This is where most of the engineering happens!

def constant_velocity_step(self, batch, batch_idx, string):
   
    x, y = prep_data_for_step(self, batch)
    features = [1, 2, 4, 5]
    x_features = x[:, :, features]
    y_hat_list = []
    for k in range(self.future_sequence_length):
        y_hat_k = self(x_features)
        y_hat_list.append(y_hat_k)
        if y_hat_k.dim() < 3:
            y_hat_k = y_hat_k.unsqueeze(1)
        x_features = torch.cat([x_features[:, 1:, :], y_hat_k], dim=1)

    y_hat = torch.stack(y_hat_list, dim=1).squeeze(dim=2)
    y_compare = y[:, :, 1:3]
    y_hat_compare = y_hat[:, :, 0:2]
    loss = F.mse_loss(y_hat_compare, y_compare)
    self.log(f"{string}_loss", loss)
    return loss


def mlp_step(self, batch, batch_idx, string):
   
    x, y = prep_data_for_step(self, batch)
    y_hat_list = []
    for k in range(self.future_sequence_length):
        y_hat_k = self(x)
        y_hat_list.append(y_hat_k)
        if y_hat_k.dim() < 3:
            y_hat_k = y_hat_k.unsqueeze(1)
        x = torch.cat([x[:, 1:, :], y_hat_k], dim=1)

    y_hat = torch.stack(y_hat_list, dim=1).squeeze()
    loss = F.mse_loss(y_hat, y)
    self.log(f"{string}_loss", loss)
    return loss


def lstm_step(self, batch, batch_idx, string):

    # Prepare data for step
    x, y = prep_data_for_step(self,batch)
    
    y_hat_list = []
    for k in range(self.future_sequence_length):
        # Forward pass
        y_hat_k = self(x)
        y_hat_list.append(y_hat_k)

        # Prepare input for next step
        if y_hat_k.dim() < 3:
            y_hat_k = y_hat_k.unsqueeze(1)
        x = torch.cat([x[:, 1:, :], y_hat_k], dim=1)
    
    # Stack predictions and compute loss
    y_hat = torch.stack(y_hat_list, dim=1).squeeze()
    loss = F.mse_loss(y_hat, y)
    self.log(f"{string}_loss", loss)
    
    return loss



# TODO: This is a hacky way to load one rectangular block from the data, and divide it into x and y of different
#  sizes afterwards.
#  If you don't do it like this, you run into trouble. Just stay aware of this.

def prep_data_for_step(self, batch):
    x = batch[:, :self.past_sequence_length, :]
    y = batch[:, self.past_sequence_length:, :]
    return x, y