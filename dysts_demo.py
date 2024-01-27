from neuralforecast.losses.numpy import mae, mse
from neuralforecast.core import NeuralForecast
from neuralforecast.auto import AutoNHITS
from ray import tune
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dysts_wrapper as dw
import torch.nn.functional as F

Y_df = dw.make(seqlen=3000)

n_series = 10
all_series = Y_df.unique_id.unique()
series = np.random.choice(all_series, n_series, replace=False)
series = series.tolist()
Y_df = Y_df[Y_df.unique_id.isin(series)]

# For this excercise we are going to take 20% of the DataSet
n_time = len(Y_df.ds.unique())
val_size = int(0.2 * n_time)
test_size = int(0.2 * n_time)

H = 100
n_step = 3
alpha = 5
L = alpha * H
W = (alpha + 1) * H

# Use your own config or AutoNHITS.default_config
nhits_config = {
    "step_size": tune.choice([3]),
    # Initial Learning rate
    "learning_rate": tune.choice([1e-3]),
    # Number of SGD steps
    "max_steps": tune.choice([1000]),
    # input_size = multiplier * H
    "input_size": tune.choice([L * n_step]),
    "batch_size": tune.choice([n_series]),
    # Number of windows in batch
    "windows_batch_size": tune.choice([256]),
    # MaxPool's Kernelsize
    "n_pool_kernel_size": tune.choice([[2, 2, 2]]),
    # Interpolation expressivity ratios
    "n_freq_downsample": tune.choice([[24, 12, 1]]),
    # Type of non-linear activation
    "activation": tune.choice(["ReLU"]),
    # Blocks per each 3 stacks
    "n_blocks": tune.choice([[1, 1, 1]]),
    # 2 512-Layers per block for each stack
    "mlp_units": tune.choice([[[512, 512], [512, 512], [512, 512]]]),
    # Type of multi-step interpolation
    "interpolation_mode": tune.choice(["linear"]),
    # Compute validation every 100 epochs
    "val_check_steps": tune.choice([100]),
    "random_seed": tune.randint(1, 10),
}

models = [AutoNHITS(h=H * n_step, config=nhits_config, num_samples=1)]

nf = NeuralForecast(models=models, freq=1)

Y_hat_df = nf.cross_validation(
    df=Y_df, val_size=val_size, test_size=test_size, n_windows=None, step_size=n_step
)

print(nf.models[0].results.get_best_result().config)

y_true = Y_hat_df.y.values
y_hat = Y_hat_df["AutoNHITS"].values

print("MAE: ", mae(y_hat, y_true))
print("MSE: ", mse(y_hat, y_true))

y_true = y_true.reshape(n_series, -1, H, n_step)
y_hat = y_hat.reshape(n_series, -1, H, n_step)
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

        # xv = np.arange((w_idx + H) * n_step)
        xv = np.arange(H * n_step)
        # ytrue = np.empty((w_idx + H, n_step))

        # for i in range(w_idx):
        #    ytrue[i] = y_true[series_idx, i, 0]

        # ytrue[w_idx:] = y_true[series_idx, w_idx]
        ytrue = y_true[series_idx, w_idx]

        # yhat = np.empty((H + 1, n_step))
        # yhat[0] = ytrue[-(H + 1)]
        # yhat[1:] = y_hat[series_idx, w_idx, :]
        yhat = y_hat[series_idx, w_idx, :]

        fig = plt.figure(figsize=plt.figaspect(2.0))

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(xv, ytrue.reshape(-1), label="True")
        ax.plot(xv[-n_step * (H + 1) :], yhat.reshape(-1), label="Forecast")
        ax.grid(True)

        ax = fig.add_subplot(2, 1, 2, projection="3d")
        ax.plot(*ytrue.T, label="True")
        ax.plot(*yhat.T, label="Forecast")

        plt.legend()
        plt.show()
        plt.close()

    except Exception as e:
        print(e)
