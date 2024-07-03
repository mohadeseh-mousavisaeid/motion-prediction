import torch
import torch.nn as nn


class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):

        x_center = x[:, 1]  # xCenter
        y_center = x[:, 2]  # yCenter
        x_velocity = x[:, 4]  # xVelocity
        y_velocity = x[:, 5]  # yVelocity
        
        # old position + velocity * dt
        new_x_center = x_center + self.dt * x_velocity
        new_y_center = y_center + self.dt * y_velocity
        
        return torch.stack((new_x_center, new_y_center), dim=1)    

    
    
class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=1.0):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt
    
    def forward(self, x):
        x_center = x[:, 1]  # xCenter
        y_center = x[:, 2]  # yCenter
        x_velocity = x[:, 4]  # xVelocity
        y_velocity = x[:, 5]  # yVelocity
        x_acceleration = x[:, 6]  # xAcceleration
        y_acceleration = x[:, 7]  # yAcceleration

        # old position + velocity * dt + 0.5 * acceleration * dt^2
        new_x_center = x_center + x_velocity * self.dt + 0.5 * x_acceleration * self.dt**2
        new_y_center = y_center + y_velocity * self.dt + 0.5 * y_acceleration * self.dt**2

        return torch.stack((new_x_center, new_y_center), dim=1)
    

# Single Track Model (Bicycle Model)
class SingleTrackModel(nn.Module):
    def __init__(self, dt=1.0):
        super(SingleTrackModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        x_center = x[:, 1]  # xCenter
        y_center = x[:, 2]  # yCenter
        heading = x[:, 3]  # heading
        x_velocity = x[:, 4]  # xVelocity
        y_velocity = x[:, 5]  # yVelocity
        
        # Calculate the change in position considering the heading
        delta_x = (x_velocity * torch.cos(heading) - y_velocity * torch.sin(heading)) * self.dt
        delta_y = (x_velocity * torch.sin(heading) + y_velocity * torch.cos(heading)) * self.dt

        # For single track model, the new position is old position plus the calculated delta
        new_x_center = x_center + delta_x
        new_y_center = y_center + delta_y

        # Return the new positions as the output
        return torch.stack((new_x_center, new_y_center), dim=1)
