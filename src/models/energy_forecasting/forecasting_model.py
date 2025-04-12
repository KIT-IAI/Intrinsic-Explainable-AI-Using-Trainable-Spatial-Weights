import os
from typing import Optional

import torch
import torch.nn.functional as F
import lightning as L

import matplotlib.pyplot as plt


class ForecastingModel(L.LightningModule):
    """
    Basic neural network class to implement methods all havin in common.
    """

    def __init__(self,
                 encoder: torch.nn.Module,
                 fusioner: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr: float,
                 forecast_horizon: int,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                 compile: bool = False,
                 debug: bool = False,
                 output_dir: str = '.'):
        """Initialize dataset, data loader, network dimensions, and torch layers.

        :param encoder: Endoder network for energy, weather, and calendar input.
        :param fusioner: Fusioner for the latent energy, weather, and calendar data
            that returns the final energy forecast.
        :param optimizer: Optimizer to use for training.
        :param scheduler: Scheduler to use for training.
        :param forecast_horizon: Energy forecast horizon.
        :param compile: Enable torch.compile() on the model, defaults to False.
        :param debug: Enable debugging for more detailed logs, defaults to False.
        :param output_dir: Output directory to use for the logs, defaults to '.'.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)  # , ignore=['encoder', 'fusioner']

        self.encoder = self.hparams.encoder
        self.fusioner = self.hparams.fusioner

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.predict_step_outputs = []
        self.forecast_shape = (1, self.hparams.forecast_horizon)

    def setup(self, stage: str):
        if self.hparams.compile and stage == 'fit':
            self.encoder = torch.compile(self.hparams.encoder)
            self.fusioner = torch.compile(self.hparams.fusioner)

    def forward(self, batch: torch.Tensor, prediction=False):
        """ Inference of the neural network model. """
        timesteps, energy, weather, calendar = batch

        (
            latent_energy,
            latent_weather,
            latent_calendar
        ) = self.encoder(energy, weather, calendar, prediction=prediction)

        y_hats = self.fusioner(latent_energy, latent_weather, latent_calendar,
                               input=(energy, weather, calendar),
                               timesteps=timesteps, prediction=prediction)
        assert len(y_hats.shape) == 4

        return y_hats, latent_energy, latent_weather, latent_calendar

    def loss(self, y, y_hats):
        assert len(y.shape) == 3
        assert len(y_hats.shape) == 4
        assert y.shape[1] == 1

        sum_loss = 0
        n_forecasts = y_hats.shape[2]
        loss_method = F.l1_loss
        for i in range(n_forecasts):
            y_hat = y_hats[:, [0], [i]]
            loss = loss_method(y, y_hat)
            sum_loss += loss

        # return sum_loss / n_forecasts
        return sum_loss

    def configure_optimizers(self):
        """ Configure optimizers and scheduler to return for lightning. """
        optimizer = self.hparams.optimizer(self.trainer.model.parameters(),
                                           self.hparams.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        """ Perform training stel given a batch. """
        x, y = batch
        y_hats, latent_energy, latent_weather, latent_calendar = self(x)
        loss = self.loss(y, y_hats)

        output = {
            'y': y.detach().cpu(),
            'y_hat': y_hats.detach().cpu(),
            'loss': loss,
        }
        self.training_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        """ Calculate loss mean after epoch and for logging/evaluation. """
        outputs = self.training_step_outputs
        with torch.no_grad():
            y = torch.cat([x['y'] for x in outputs])
            y_hat = torch.cat([x['y_hat'] for x in outputs], axis=0)
            self.log('loss', self.loss(y, y_hat), prog_bar=True)

            # calculate loss
            metric_dict = {}
            metric_dict[f'train/mae'] = F.l1_loss(y_hat[:, :, -1], y)
            metric_dict[f'train/mse'] = F.mse_loss(y_hat[:, :, -1], y)

            for key in metric_dict:
                self.log(key, metric_dict[key])

        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """ Perform validation step given a validation batch. """
        x, y = batch
        y_hats, latent_energy, latent_weather, latent_calendar = self(x)

        loss = self.loss(y, y_hats)

        output = {
            'y': y.cpu(),
            'y_hat': y_hats.cpu(),
            'val_loss': loss,
        }
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        """ Calculate suitable metrics on validation set for logging and evaulation. """
        outputs = self.validation_step_outputs
        with torch.no_grad():
            y = torch.cat([x['y'] for x in outputs])
            y_hat = torch.cat([x['y_hat'] for x in outputs], axis=0)
            self.log('val_loss', self.loss(y, y_hat), prog_bar=True)

            losses = []
            for i in range(y.shape[2]):
                loss = self.loss(y[..., [i]], y_hat[..., [i]])
                loss = loss.numpy().item()
                losses.append(loss)
            plt.plot(losses)
            plt.title('val loss')
            plt.xlabel('Forecast Horizon [h]')
            plt.savefig(os.path.join(self.hparams.output_dir, 'val_loss.png'))
            plt.close()

            # plt.hist(y_hat.flatten() - y.flatten(), )

            # calculate loss
            metric_dict = {}
            for i in range(y_hat.shape[2]):
                metric_dict[f'val_loss_{i}'] = F.l1_loss(y_hat[:, :, i], y)

            for key in metric_dict:
                self.log(key, metric_dict[key], prog_bar=True)

            if self.hparams.debug:
                self.encoder.debug(outputs, self.hparams.output_dir)
                self.fusioner.debug(outputs, self.hparams.output_dir)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        """ Perform validation step given a validation batch. """
        x = batch
        y_hats, latent_energy, latent_weather, latent_calendar = self(x, prediction=True)

        output = {
            'y_hat': y_hats.cpu(),
        }
        self.predict_step_outputs.append(output)

        return output

    def on_predict_epoch_end(self):
        """ Calculate suitable metrics on validation set for logging and evaulation. """
        outputs = self.predict_step_outputs

        if self.hparams.debug:
            self.encoder.debug(outputs, self.hparams.output_dir, detailed=True)
            self.fusioner.debug(outputs, self.hparams.output_dir, detailed=True)

        self.predict_step_outputs = []
