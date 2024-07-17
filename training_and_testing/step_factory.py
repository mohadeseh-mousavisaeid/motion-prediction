from enums.model import Model
from step.steps import mlp_step, constant_velocity_step, lstm_step

class StepFactory:

    def get_step(self,batch, batch_idx, string, model:str):
        
        if model== Model.CONSTANT_VELOCITY.value:
            return constant_velocity_step(self,batch, batch_idx, string)
        
        elif model== Model.MLP.value:
            return mlp_step(self,batch, batch_idx, string)
    
        elif model== Model.LSTM.value:
            return lstm_step(self,batch, batch_idx, string)