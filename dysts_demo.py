from neuralforecast.losses.numpy import mae, mse
from neuralforecast.core import NeuralForecast
from neuralforecast.auto import AutoNHITS
import ray
from ray import tune
import matplotlib.pyplot as plt
import numpy as np
import dysts_wrapper as dw
from config import get_config
import argparse
from datetime import datetime

datestr = datetime.now().strftime("%Y%m%d%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
args = parser.parse_args()
cfgyml = get_config(args.cfg)

Y_df = dw.make(seqlen=cfgyml.seqlen)

n_series = cfgyml.n_series
all_series = Y_df.unique_id.unique()
series = np.random.choice(all_series, n_series, replace=False)
series = series.tolist()
print(f"Selected series: {', '.join(series)}")

Y_df = Y_df[Y_df.unique_id.isin(series)]

n_time = len(Y_df.ds.unique())
val_size = int(cfgyml.test_percent * n_time)
test_size = int(cfgyml.test_percent * n_time)

H = cfgyml.H
step_size = 3  # 3d data
alpha = cfgyml.alpha
L = alpha * H
W = (alpha + 1) * H

if cfgyml.lowmem:
    batch_size = cfgyml.batch_size
    windows_batch_size = [0]
else:
    batch_size = [n_series]
    windows_batch_size = cfgyml.batch_size

nhits_config = {
    "step_size": tune.choice([step_size]),
    # Initial Learning rate
    "learning_rate": tune.choice(cfgyml.learning_rate),
    # Number of SGD steps
    "max_steps": tune.choice(cfgyml.max_steps),
    # input_size = multiplier * H
    "input_size": tune.choice([L * step_size]),
    "batch_size": tune.choice(batch_size),
    # Number of windows in batch
    "windows_batch_size": tune.choice(windows_batch_size),
    # MaxPool's Kernelsize
    "n_pool_kernel_size": tune.choice(cfgyml.n_pool_kernel_size),
    # Interpolation expressivity ratios
    "n_freq_downsample": tune.choice(cfgyml.n_freq_downsample),
    # Type of non-linear activation
    "activation": tune.choice(["ReLU"]),
    # Blocks per each 3 stacks
    "n_blocks": tune.choice(cfgyml.n_blocks),
    # 2 512-Layers per block for each stack
    "mlp_units": tune.choice(cfgyml.mlp_units),
    # Type of multi-step interpolation
    "interpolation_mode": tune.choice(["linear"]),
    # Compute validation every N epochs
    "val_check_steps": tune.choice(cfgyml.val_check_steps),
    "random_seed": tune.randint(1, 10),
    "lowmem": tune.choice([cfgyml.lowmem]),
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

nf = NeuralForecast(models=models, freq=1, lowmem=cfgyml.lowmem)

Y_hat_df = nf.cross_validation(
    df=Y_df, val_size=val_size, test_size=test_size, n_windows=None, step_size=step_size
)

ray.shutdown()

best_config = nf.models[0].results.get_best_result().config
print(best_config)

y_true = Y_hat_df.y.values
y_hat = Y_hat_df["AutoNHITS"].values

print("MAE: ", mae(y_hat, y_true))
print("MSE: ", mse(y_hat, y_true))

y_true = y_true.reshape(n_series, -1, H, step_size)
y_hat = y_hat.reshape(n_series, -1, H, step_size)

datafile = f"datafiles/series{n_series}-step{best_config['max_steps']}-{datestr}.npy"
np.save(
    datafile,
    {"series": series, "config": best_config, "y_true": y_true, "y_hat": y_hat},
    allow_pickle=True,
)

nwindow = y_true.shape[1]
wrange = (0, nwindow - 1)
while True:
    try:
        sname = input(f"enter series {series} (q to quit): ")
        if sname == "q":
            break
        series_idx = series.index(sname)
        w_idx = input(f"enter window index [{wrange[0],wrange[1]}] (q to quit): ")
        if w_idx == "q":
            break
        w_idx = int(w_idx)
        if not (wrange[0] <= w_idx and w_idx <= wrange[1]):
            raise Exception("out of range")

        # xv = np.arange((w_idx + H) * step_size)
        xv = np.arange(H * step_size)
        # ytrue = np.empty((w_idx + H, step_size))

        # for i in range(w_idx):
        #    ytrue[i] = y_true[series_idx, i, 0]

        # ytrue[w_idx:] = y_true[series_idx, w_idx]
        ytrue = y_true[series_idx, w_idx]

        # yhat = np.empty((H + 1, step_size))
        # yhat[0] = ytrue[-(H + 1)]
        # yhat[1:] = y_hat[series_idx, w_idx, :]
        yhat = y_hat[series_idx, w_idx, :]

        fig = plt.figure(figsize=plt.figaspect(2.0))

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(xv, ytrue.reshape(-1), label="True")
        ax.plot(xv[-step_size * (H + 1) :], yhat.reshape(-1), label="Forecast")
        ax.grid(True)

        ax = fig.add_subplot(2, 1, 2, projection="3d")
        ax.plot(*ytrue.T, label="True")
        ax.plot(*yhat.T, label="Forecast")

        plt.legend()
        plt.show()
        plt.close()

    except Exception as e:
        print(e)
