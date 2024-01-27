from neuralforecast.losses.numpy import mae, mse
from neuralforecast.core import NeuralForecast
from neuralforecast.auto import AutoNHITS
from ray import tune
import matplotlib.pyplot as plt
import numpy as np
import dysts_wrapper as dw


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--legacy",
    action="store_true",
    default=False,
    help="original nixtla implementation",
)
parser.add_argument(
    "--seqlen", default=3000, type=int, help="training data sequence length per series"
)
parser.add_argument(
    "--n_series",
    default=10,
    type=int,
    help="number of chaotic series to include in dataset",
)
parser.add_argument("--H", default=100, type=int, help="horizon")
parser.add_argument("--alpha", default=5, type=int, help="input_size=alpha*H")
parser.add_argument(
    "--test_percent", default=0.2, type=float, help="percent for test and val"
)
parser.add_argument("--num_samples", default=1, type=int, help="cv training samples")
parser.add_argument("--batch_size", default=256, type=int, help="training batch size")
parser.add_argument(
    "--max_steps", default=1000, type=int, help="number of training steps per model"
)
args = parser.parse_args()

Y_df = dw.make(seqlen=args.seqlen)

n_series = args.n_series
all_series = Y_df.unique_id.unique()
series = np.random.choice(all_series, n_series, replace=False)
series = series.tolist()
print("Selected series:", series)

Y_df = Y_df[Y_df.unique_id.isin(series)]

# For this excercise we are going to take 20% of the DataSet
n_time = len(Y_df.ds.unique())
val_size = int(args.test_percent * n_time)
test_size = int(args.test_percent * n_time)

H = args.H
n_step = 3
alpha = args.alpha
L = alpha * H
W = (alpha + 1) * H

if args.legacy:
    batch_size = n_series
    windows_batch_size = args.batch_size
else:
    batch_size = args.batch_size
    windows_batch_size = 0

# Use your own config or AutoNHITS.default_config
nhits_config = {
    "step_size": tune.choice([n_step]),
    # Initial Learning rate
    "learning_rate": tune.choice([1e-3]),
    # Number of SGD steps
    "max_steps": tune.choice([args.max_steps]),
    # input_size = multiplier * H
    "input_size": tune.choice([L * n_step]),
    "batch_size": tune.choice([batch_size]),
    # Number of windows in batch
    "windows_batch_size": tune.choice([windows_batch_size]),
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
    # Compute validation every N epochs
    "val_check_steps": tune.choice([1000]),
    "random_seed": tune.randint(1, 10),
    "lowmem": tune.choice([args.legacy == False]),
}

models = [
    AutoNHITS(
        h=H * n_step,
        input_size=L * n_step,
        step_size=n_step,
        config=nhits_config,
        num_samples=args.num_samples,
    )
]

nf = NeuralForecast(models=models, freq=1, lowmem=args.legacy == False)

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
