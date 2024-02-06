import argparse
from datetime import datetime
import json

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

Y_df = pd.read_csv(f"datasets/{cfgyml.dataset}.csv")
with open(f"datasets/{cfgyml.dataset}.json") as f:
    dsmd = json.load(f)

series = Y_df.unique_id.unique()
series = series.tolist()


H = cfgyml.H
data_dim = dsmd["ndim"]
alpha = cfgyml.alpha
L = alpha * H
W = L + H

nhits_config = {
    "step_size": tune.choice([data_dim]),
    # Initial Learning rate
    "learning_rate": tune.grid_search(cfgyml.learning_rate),
    # Number of SGD steps
    "max_steps": tune.grid_search(cfgyml.max_steps),
    "lr_decay_gamma": tune.grid_search(cfgyml.lr_decay_gamma),
    "num_lr_decays": tune.grid_search(cfgyml.num_lr_decays),
    # input_size = multiplier * H
    "input_size": tune.choice([L * data_dim]),
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
        h=H * data_dim,
        input_size=L * data_dim,
        step_size=data_dim,
        config=nhits_config,
        num_samples=cfgyml.num_samples,
    )
]

nf = NeuralForecast(models=models, freq=100, lowmem=True)

Y_hat_df = nf.cross_validation(
    df=Y_df,
    n_series_val=cfgyml.n_series_val,
    n_series_test=cfgyml.n_series_test,
    step_size=data_dim,
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
y_true = y_true.reshape(n_series, -1, H, data_dim)
y_hat = y_hat.reshape(n_series, -1, H, data_dim)

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
