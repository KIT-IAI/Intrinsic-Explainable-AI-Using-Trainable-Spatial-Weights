import os

import numpy as np
import xarray as xr

from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            mean_absolute_percentage_error, r2_score

import matplotlib.pyplot as plt


def get_metrics(cfg, object_dict, energy_datasets):
    """Calculate and return all metrics needed for evaluation.

    :param cfg: Hydra run config.
    :param object_dict: Dict containing the model, trainer, and forecasts.
    :param energy_datasets: The energy datasets.
    :return: Returns the calculated metrics within the object dict.
    """
    metric_dict = {}
    for key in ['train', 'val', 'test']:
        energy = energy_datasets[key]
        ground_truth = np.array([x[1] for x in energy])
        y = xr.Dataset(
            data_vars=dict(
                [(
                    target.replace('_target', ''),
                    (['time', 'forecast'], ground_truth[:, i])
                 )
                 for i, target in enumerate(cfg.targets)]
            ),
            coords= {
                'time': energy.time,
                'forecast': range(1, cfg.forecast_horizon + 1)
            }
        )
        y_hat = object_dict[key]
        for target in cfg.targets:
            if len(cfg.targets) == 1:
                prefix = key
            else:
                prefix = f'{target}/{key}'
            y_target = energy.inverse_normalize_energy(y[target], target)
            y_hat_target = energy.inverse_normalize_energy(y_hat[target], target)
            y_hat_target = y_hat_target.where(y_hat_target >= 0, 0)
            d = calculate_metrics(prefix, y_target, y_hat_target,
                            plot_dir=cfg.paths.output_dir)
            metric_dict.update(d)
    return metric_dict


def calculate_metric_hourly(metric, y, y_hat):
    """ Hourly metric calculation helper. """
    hourly_metric_values = []
    for i in range(y_hat.shape[1]):
        value = metric(y_true=y[:, i], y_pred=y_hat[:, i])
        hourly_metric_values.append(value)
    return np.array(hourly_metric_values)


def calculate_metrics(prefix, y, y_hat, hourly=True, plot_dir='.'):
    """ Metric calculation helper. """
    logs = {}
    metrics = [
        ('ME', lambda y_true, y_pred:
            np.mean(y_true - y_pred)),
        ('MAE', lambda y_true, y_pred:
            mean_absolute_error(y_true, y_pred)),
        ('RMSE', lambda y_true, y_pred:
            np.sqrt(mean_squared_error(y_true, y_pred))),
        ('MAPE', lambda y_true, y_pred:
            mean_absolute_percentage_error(y_true, y_pred) * 100),
        ('R2', lambda y_true, y_pred:
            r2_score(y_true, y_pred)),
    ]
    n_forecasts = y_hat.shape[1]
    forecast_horizon = y_hat.shape[2]
    for name, metric in metrics:
        for i in range(n_forecasts):
            value = metric(y_true=y, y_pred=y_hat[:, i])
            key = f'{prefix}/scaled/{name}/{i}'
            logs[f'{key}/all'] = value
            hourly_metrics = calculate_metric_hourly(metric, y, y_hat[:, i])
            if hourly:
                for j in range(forecast_horizon):
                    logs[f'{key}/{j + 1:03}h'] = hourly_metrics[j]
                if plot_dir:
                    plt.plot(np.arange(1, forecast_horizon + 1), hourly_metrics, label=f'{i}')
        plt.xticks(np.arange(0, forecast_horizon + 1, 24))
        plt.gca().set_axisbelow(True)
        plt.gca().xaxis.grid(color='gray', linestyle='dashed')
        plt.gca().yaxis.grid(color='gray', linestyle='dashed')
        plt.xlabel('Forcasting Hour')
        plt.ylabel(name)
        plt.savefig(os.path.join(plot_dir, f'{prefix}_{name}.png'))
        plt.title(f'{prefix}_{name}')
        plt.close()

        value = metric(y_true=y, y_pred=y_hat[:, -1])
        logs[f'{prefix}/scaled/{name}'] = value

    return logs