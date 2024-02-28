import argparse
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
import ray
from ray import tune

from config import get_config
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
# from neuralforecast.losses.numpy import mae, mse


def n_unique_configs(config):
    n_per_param = []
    for k, v in config.items():
        if type(v) == dict:
            n_per_param.append(len(v["grid_search"]))
        elif type(v) == ray.tune.search.sample.Categorical:
            n_per_param.append(len(v))
    return np.product(n_per_param)


def get_single_config(config):
    single = {}
    for k, v in config.items():
        if type(v) == dict:
            single[k] = v["grid_search"][0]
        elif type(v) in [
            ray.tune.search.sample.Categorical,
            ray.tune.search.sample.Integer,
        ]:
            single[k] = v.sample()
        else:
            single[k] = v
    return single


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

with open(f"datasets/{cfgyml.dataset}.json") as f:
    dsmd = json.load(f)

data_dim = dsmd["ndim"]
total_pps = dsmd["points_per_series"]
pps = cfgyml.points_per_series

skiprows = lambda idx: idx != 0 and (idx - 1) % (data_dim * total_pps) >= (
    data_dim * pps
)
Y_df = pd.read_csv(
    f"datasets/{cfgyml.dataset}.csv",
    skiprows=skiprows,
)
series = Y_df.unique_id.unique().tolist()

H = cfgyml.H
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
    "num_workers_loader": tune.choice([int(os.cpu_count() / 2)]),
}

best_config = None
if n_unique_configs(nhits_config) == 1:
    best_config = get_single_config(nhits_config)

models = [
    AutoNHITS(
        h=H * data_dim,
        input_size=L * data_dim,
        step_size=data_dim,
        config=nhits_config,
        num_samples=cfgyml.num_samples,
        best_config=best_config,
    )
]

nf = NeuralForecast(models=models, freq=100, lowmem=True)

yh_df, yt_df = nf.cross_validation(
    df=Y_df,
    n_series_val=cfgyml.n_series_val,
    n_series_test=cfgyml.n_series_test,
    step_size=data_dim,
)
ray.shutdown()

if best_config is None:
    best_config = nf.models[0].results.get_best_result().config
print(best_config)
del nf

yh_raw = yh_df.AutoNHITS.to_numpy().astype(np.float32)
del yh_df
y_hat = yh_raw.reshape(cfgyml.n_series_test, pps - W + 1, H, data_dim)

yt_raw = yt_df.y.to_numpy().astype(np.float32)
del yt_df
yt_raw = yt_raw.reshape(cfgyml.n_series_test, pps, data_dim)
yt_raw = yt_raw[:, L:, :]
unfold = torch.nn.Unfold((H, 1))
yt_raw_th = torch.from_numpy(yt_raw)
yt_raw_th = yt_raw_th.permute(0, 2, 1)
yt_th = unfold(yt_raw_th[:, :, :, None]).permute(0, 2, 1)
del yt_raw_th
yt_th = yt_th.reshape(cfgyml.n_series_test, -1, data_dim, H).permute(0, 1, 3, 2)
y_true = yt_th.numpy()

# print("MAE: ", mae(y_hat, y_true))
# print("MSE: ", mse(y_hat, y_true))

if args.save:
    datafile = f"predictions/{args.fn}.npy"
    np.save(
        datafile,
        {
            "config": best_config,
            "y_true": y_true,
            "y_hat": y_hat,
            "dataset": cfgyml.dataset,
        },
        allow_pickle=True,
    )
