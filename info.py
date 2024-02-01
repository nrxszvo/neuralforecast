import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def load(fn):
    return np.load(fn, allow_pickle=True).item()


def get_ys(d):
    yt = d["y_true"][0]
    yh = d["y_hat"][0]
    return yt, yh


def calc_mae(d, ax=(0, 1, 2)):
    yt, yh = get_ys(d)
    mae = np.abs(yt - yh)
    if len(ax) > 0:
        mae = mae.mean(axis=ax)
    return mae


def calc_smape(d, ax=(0, 1, 2)):
    yt, yh = get_ys(d)
    t = np.prod([yt.shape[i] for i in ax])
    smape = (2 / t) * np.sum(np.abs(yt - yh) / (np.abs(yt) + np.abs(yh)), axis=ax)
    return smape


def plot_dist(ds):
    ax1 = plt.figure().add_subplot()
    ax2 = plt.figure().add_subplot()
    for d in ds:
        err = calc_smape(d, ax=(1, 2))
        mu, std = norm.fit(err)
        ax1.hist(err, bins=100, density=True, alpha=0.6, label=d["series"][0])
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax1.plot(x, p, "k", linewidth=2)
        ax2.plot(err, label=d["series"][0])

    ax1.set_title(", ".join([d["series"][0] for d in ds]) + "-- error distribution")
    ax2.set_title("Error by window")
    plt.show()
    plt.close()


def plot_summary(ds):
    names = []
    errs = []
    for d in ds:
        names.append(d["series"][0])
        errs.append(calc_smape(d))
    plt.bar(names, errs)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_summary_menu(available):
    inp = input("plot all or select? [a|s]: ")
    if inp == "a":
        plot_summary([load(fn) for name, fn in available])
    else:
        plot_choices_menu(available, plot_summary)


def plot_3d(d, idx):
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(*d["y_true"][0, idx].T, label="y_true")
    ax.plot(*d["y_hat"][0, idx].T, label="y_hat")
    plt.show()
    plt.close()


def print_hparams(d):
    print(d["series"][0])
    print(f'\tlearning rate: {d["config"]["learning_rate"]}')
    print(f'\tkernel size: {d["config"]["n_pool_kernel_size"]}')
    print(f'\tdownsample: {d["config"]["n_freq_downsample"]}')
    print(f'\tmlp units: {d["config"]["mlp_units"]}')


def collect_available(pattern, dirname):
    available = []
    for fn in sorted(os.listdir(dirname)):
        m = re.match(pattern, fn)
        if m != None:
            available.append((m.group(1), f"{dirname}/{fn}"))
    return available


def plot_choices_menu(available, plot_fn):
    choices = []
    while True:
        inp = input("idx to add, p to plot, q to quit: ")
        if inp == "q":
            break
        elif inp == "p":
            plot_fn([load(available[ch][1]) for ch in choices])
            choices = []
        else:
            try:
                choices.append(int(inp))
            except Exception as e:
                print(e)


def plot_dist_menu(available):
    plot_choices_menu(available, plot_dist)


def plot_3d_menu(available):
    while True:
        try:
            sidx = input("series idx to plot, q to quit: ")
            if sidx == "q":
                break
            else:
                try:
                    d = load(available[int(sidx)][1])
                    nwin = d["y_true"].shape[1]
                    while True:
                        widx = input(f"window to plot [0, {nwin}] (q to quit): ")
                        if widx == "q":
                            break
                        else:
                            plot_3d(d, int(widx))
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("opt", choices=["dist", "smape", "3d"], help="plot type")
    parser.add_argument(
        "--pattern", default="(.*).npy", help="re for matching npy files"
    )
    parser.add_argument("--dirname", default="datafiles", help="data file directory")
    args = parser.parse_args()

    available = collect_available(args.pattern, args.dirname)

    for i, (name, fn) in enumerate(available):
        print(f"{i}: {name}")

    if args.opt == "dist":
        plot_dist_menu(available)
    elif args.opt == "3d":
        plot_3d_menu(available)
    elif args.opt == "smape":
        plot_summary_menu(available)
