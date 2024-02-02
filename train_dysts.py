import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import ray
from ray import tune

from config import get_config
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.numpy import mae, mse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
parser.add_argument(
    "--save", default=False, action="store_true", help="save prediction data"
)
parser.add_argument(
    "--fn",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="prediction file name",
)

args = parser.parse_args()
cfgyml = get_config(args.cfg)

Y_df = pd.read_csv(cfgyml.dataset)

series = Y_df.unique_id.unique()
series = series.tolist()

H = cfgyml.H
step_size = 3  # 3d data
alpha = cfgyml.alpha
L = alpha * H
W = (alpha + 1) * H

nhits_config = {
    "step_size": tune.choice([step_size]),
    # Initial Learning rate
    "learning_rate": tune.grid_search(cfgyml.learning_rate),
    # Number of SGD steps
    "max_steps": tune.grid_search(cfgyml.max_steps),
    # input_size = multiplier * H
    "input_size": tune.choice([L * step_size]),
    "batch_size": tune.grid_search(cfgyml.batch_size),
    "stack_types": tune.grid_search(cfgyml.stack_types),
    # MaxPool's Kernelsize
    "n_pool_kernel_size": tune.grid_search(cfgyml.n_pool_kernel_size),
    # Interpolation expressivity ratios
    "n_freq_downsample": tune.grid_search(cfgyml.n_freq_downsample),
    # Type of non-linear activation
    "activation": tune.grid_search(["ReLU"]),
    # Blocks per each 3 stacks
    "n_blocks": tune.grid_search(cfgyml.n_blocks),
    # 2 512-Layers per block for each stack
    "mlp_units": tune.grid_search(cfgyml.mlp_units),
    # Type of multi-step interpolation
    "interpolation_mode": tune.choice(["linear"]),
    # Compute validation every N epochs
    "val_check_steps": tune.choice(cfgyml.val_check_steps),
    "random_seed": tune.randint(1, 10),
    "lowmem": tune.choice([True]),
}

models = [
    AutoNHITS(
        h=H * step_size,
        input_size=L * step_size,
        step_size=step_size,
        config=nhits_config,
        num_samples=cfgyml.num_samples,
    )
]

nf = NeuralForecast(models=models, freq=100, lowmem=True)

Y_hat_df = nf.cross_validation(
    df=Y_df,
    n_series_val=cfgyml.n_series_val,
    n_series_test=cfgyml.n_series_test,
    step_size=step_size,
)

ray.shutdown()

best_config = nf.models[0].results.get_best_result().config
print(best_config)

y_true = Y_hat_df.y.values
y_hat = Y_hat_df["AutoNHITS"].values

print("MAE: ", mae(y_hat, y_true))
print("MSE: ", mse(y_hat, y_true))

available_series = Y_hat_df.index.unique().to_numpy()
n_series = len(available_series)
y_true = y_true.reshape(n_series, -1, H, step_size)
y_hat = y_hat.reshape(n_series, -1, H, step_size)

if args.save:
    datafile = f"predictions/{args.fn}.npy"
    np.save(
        datafile,
        {
            "series": available_series,
            "config": best_config,
            "y_true": y_true,
            "y_hat": y_hat,
            "results": nf.models[0].results,
        },
        allow_pickle=True,
    )
