import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class LitModule(pl.LightningModule):
    
    def __init__(self, model, number_of_features, sequence_length, past_sequence_length, future_sequence_length, batch_size):
        super().__init__()
        self.model = model
        self.nx = number_of_features
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.batch_size = batch_size


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        string = "training"
        loss = self.step(batch, batch_idx, string)
        return loss

    def validation_step(self, batch, batch_idx):
        string = "validation"
        loss = self.step(batch, batch_idx, string)
        return loss

    def test_step(self, batch, batch_idx):
        string = "test"
        loss = self.step(batch, batch_idx, string)
        return loss

    def step(self, batch, batch_idx, string):
        """
        This is the main step function that is used by training_step, validation_step, and test_step.
        """
        # TODO: You have to modify this based on your task, model and data. This is where most of the engineering happens!
        x, y = self.prep_data_for_step(batch)
        x_acc = x
        y_hat_list = []
        for k in range(self.future_sequence_length):
            x_k = x[:,-2:,0:3]
            # augment this with your constant acceleration to get a full state vector again
            x_aug = torch.cat([x_k, x_acc], dim=1)
            y_hat_k = self(x_aug)
            y_hat_list.append(y_hat_k)
            if y_hat_k.dim() < 3:
                y_hat_k = y_hat_k.unsqueeze(1)
            
        
            x_center = x[:, :, 1]  # xCenter
            y_center = x[:, :, 2]  # yCenter

            new_x = torch.stack((x_center, y_center), dim=1)

            x = torch.cat([new_x[:, 1:, :], y_hat_k], dim=1)

        y_hat = torch.stack(y_hat_list, dim=1).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log(f"{string}_loss", loss)
        return loss
    

    def prep_data_for_step(self, batch):
        x = batch[:, :self.past_sequence_length, :]
        y = batch[:, self.past_sequence_length:, :]
        return x, y

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if parameters:
            optimizer = torch.optim.Adam(self.parameters(),
                                        lr=1e-3,
                                        weight_decay=1e-3,
                                        eps=1e-5,
                                        fused=False,
                                        amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.2,
                patience=3,
                threshold=1e-4,
                cooldown=2,
                eps=1e-6,
                verbose=True,
            )
            optimizer_and_scheduler = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "training_loss",
                    "frequency": 1,
                    "strict": True}
            }
            return optimizer_and_scheduler
        else:
            return []


