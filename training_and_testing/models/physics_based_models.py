import torch
import torch.nn as nn


class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        # Select the last time step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        # Extract features from the last time step
        x_center = x_last[:, 0]  # xCenter
        y_center = x_last[:, 1]  # yCenter
        x_velocity = x_last[:, 2]  # xVelocity
        y_velocity = x_last[:, 3]  # yVelocity
        
        # old position + velocity * dt
        new_x_center = x_center + self.dt * x_velocity
        new_y_center = y_center + self.dt * y_velocity
        new_positions = torch.stack((new_x_center, new_y_center, x_velocity, y_velocity), dim=1)  # shape: (batch_size, 4)

        # Replicate the new_positions to match the number of features
        # new_positions = new_positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # shape: (batch_size, 2, number_of_features)
        
        return new_positions
    
    
class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt
    
    def forward(self, x):
       # Select the last time step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        # Extract features from the last time step
        x_center = x_last[:, 0]  # xCenter
        y_center = x_last[:, 1]  # yCenter
        x_velocity = x_last[:, 2]  # xVelocity
        y_velocity = x_last[:, 3]  # yVelocity
        x_acceleration = x_last[:, 4]  # xAcceleration
        y_acceleration = x_last[:, 5]  # yAcceleration

        # Calculate new positions based on constant acceleration model
        new_x_center = x_center + x_velocity * self.dt + 0.5 * x_acceleration * self.dt**2
        new_y_center = y_center + y_velocity * self.dt + 0.5 * y_acceleration * self.dt**2

        # Create new positions tensor with the shape (batch_size, 2)
        new_positions = torch.stack((new_x_center, new_y_center,x_velocity, y_velocity,x_acceleration,y_acceleration), dim=1)  # shape: (batch_size, 6)
        
        # Replicate the new_positions to match the number of features
        # new_positions = new_positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # shape: (batch_size, 2, number_of_features)
        
        return new_positions

# Single Track Model (Bicycle Model)
class SingleTrackModel(nn.Module):
    def __init__(self, dt=1.0):
        super(SingleTrackModel, self).__init__()
        self.dt = dt

    def forward(self, x):

         # Select the last time step from the input sequence
        x_last = x[:, -1, :]  # shape: (batch_size, number_of_features)

        # Extract features from the last time step
        x_center = x_last[:, 1]  # xCenter
        y_center = x_last[:, 2]  # yCenter
        heading = x_last[:, 3]  # heading
        x_velocity = x_last[:, 4]  # xVelocity
        y_velocity = x_last[:, 5]  # yVelocity
        
        # Calculate the change in position considering the heading
        delta_x = (x_velocity * torch.cos(heading) - y_velocity * torch.sin(heading)) * self.dt
        delta_y = (x_velocity * torch.sin(heading) + y_velocity * torch.cos(heading)) * self.dt

        # For single track model, the new position is old position plus the calculated delta
        new_x_center = x_center + delta_x
        new_y_center = y_center + delta_y

        # Create new positions tensor with the shape (batch_size, 2)
        new_positions = torch.stack((new_x_center, new_y_center), dim=1)  # shape: (batch_size, 2)
        
        # Replicate the new_positions to match the number of features
        new_positions = new_positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # shape: (batch_size, 2, number_of_features)
        
        return new_positions
