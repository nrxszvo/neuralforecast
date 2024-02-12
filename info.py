import os
import re
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import expon


def load(fn):
    return np.load(fn, allow_pickle=True).item()


def get_ys(d):
    yt = d["y_true"]
    yh = d["y_hat"]
    return yt, yh


def calc_mae(d, ax=(0, 1, 2, 3)):
    yt, yh = get_ys(d)
    mae = np.abs(yt - yh)
    if len(ax) > 0:
        mae = mae.mean(axis=ax)
    return mae


def calc_smape(d, ax=(0, 1, 2, 3)):
    yt, yh = get_ys(d)
    t = np.prod([yt.shape[i] for i in ax])
    smape = (2 / t) * np.sum(np.abs(yt - yh) / (np.abs(yt) + np.abs(yh)), axis=ax)
    return smape


def calc_speed(yt):
    ndim = yt.ndim
    while yt.ndim < 4:
        yt = yt[None]
    dyt = np.diff(yt, axis=2)
    magdy = np.linalg.norm(dyt, axis=(3))
    nseries, nwin, winsize = magdy.shape
    nsmooth = 10
    smoother = np.ones(nsmooth)
    smoothmag = np.empty((nseries, nwin, winsize - nsmooth + 1))
    for i in range(nseries):
        for j in range(nwin):
            smoothmag[i, j] = np.convolve(magdy[i, j], smoother, mode="valid")
    speed = np.min(smoothmag, axis=2)
    idx = np.argmin(smoothmag, axis=2)
    indices = (
        np.repeat(idx[:, :, None], nsmooth, axis=2) + np.arange(nsmooth)[None, None, :]
    )
    invspeed = -(speed - speed.max())
    while indices.ndim > ndim - 1:
        indices = indices[0]
    return invspeed, indices


def plot_dist(ds):
    ax1 = plt.figure().add_subplot()
    ax2 = plt.figure().add_subplot()
    ax3 = plt.figure().add_subplot()
    # axes are: series, window, widx, dim
    for d, name in ds:
        nseries, nwin, _, _ = d["y_true"].shape
        err_win = calc_smape(d, ax=(2, 3)).reshape(-1)
        invspeed = calc_speed(d["y_true"])[0].reshape(-1)
        # mu, std = expon.fit(err)
        xax = np.arange(nseries * nwin)
        ax1.scatter(xax, err_win, s=0.5, label=name)
        ax1t = ax1.twinx()
        ax1t.scatter(
            xax, invspeed, s=0.5, alpha=0.6, label="inverse speed", color="red"
        )
        xticks = np.arange(0, nseries * nwin, nwin)
        xlabels = np.arange(nseries)
        # ax1.set_xticks(xticks, labels=xlabels)
        err_hidx = calc_smape(d, ax=(0, 1, 3))
        ax2.scatter(np.arange(len(err_hidx)), err_hidx, s=0.5, label=name)
        ax3.hist(err_win, bins=100, alpha=0.6, label=name)
        # xmin, xmax = ax1.get_xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = expon.pdf(x, mu, std)
        # ax3.plot(x, p, "k", linewidth=2)
    ax1.set_xlabel("series")
    ax1.set_ylabel("sMAPE")
    ax1t.set_ylabel("inverse speed", color="red")
    ax1.legend()
    ax1.set_title("Error by window")

    ax2.set_xlabel("horizon index")
    ax2.set_ylabel("sMAPE")
    ax2.legend()
    ax2.set_title("Error by horizon index")

    ax3.set_xlabel("sMAPE")
    ax3.set_ylabel("# windows")
    ax3.legend()
    ax3.set_title(", ".join([name for d, name in ds]) + "-- error distribution")
    plt.show()
    plt.close()


def plot_summary(ds):
    names = []
    errs = []
    for d, name in ds:
        names.append(name)
        errs.append(calc_smape(d))
    plt.bar(names, errs)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_summary_menu(available):
    for i, (name, fn) in enumerate(available):
        print(f"{i}: {name}")
    inp = input("plot all or select? [a|s]: ")
    if inp == "a":
        plot_summary([(load(fn), name) for name, fn in available])
    else:
        choices_menu(available, plot_summary)


def plot_3d(d, name, choices):
    for sidx, widx in choices:
        ax = plt.figure().add_subplot(projection="3d")
        _, highlight_idx = calc_speed(d["y_true"][sidx, widx])
        ax.plot(*d["y_true"][sidx, widx].T, label="y_true")
        ax.plot(*d["y_hat"][sidx, widx].T, label="y_hat", alpha=0.6)
        ax.plot(*d["y_true"][sidx, widx, highlight_idx].T, alpha=0.6, color="red")
        ax.legend()
    plt.title(f"{name} - {sidx} - {widx}")


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


def choices_menu(available, fn):
    for i, (name, _) in enumerate(available):
        print(f"{i}: {name}")
    choices = []
    while True:
        inp = input("idx to add (a for all): ")
        if inp == "a":
            fn([(load(fn), name) for fn, name in available])
        elif inp == "":
            if len(choices) > 0:
                fn([(load(available[ch][1]), available[ch][0]) for ch in choices])
                choices = []
            else:
                break
        else:
            try:
                choices.append(int(inp))
            except Exception as e:
                print(e)


def plot_3d_menu(ds):
    for d, name in ds:
        try:
            nser = d["y_true"].shape[0]
            nwin = d["y_true"].shape[1]
            choices = []
            while True:
                resp = input(
                    f"{name}: series and window to plot [0, {nser-1}].[0, {nwin-1}]: "
                )
                if resp == "":
                    if len(choices) > 0:
                        plot_3d(d, name, choices)
                        choices = []
                    break
                else:
                    sidx, widx = resp.split(".")
                    choices.append((int(sidx), int(widx)))
        except Exception as e:
            print(e)
    plt.show()
    plt.close()


def print_metadata(ds):
    for d, name in ds:
        mae = calc_mae(d)
        smape = calc_smape(d)
        print()
        print(name)
        print(f'\tdataset: {d["dataset"]}')
        print(f"\tMAE: {mae:.3f}")
        print(f"\tsMAPE: {smape:.3f}")
        print("\tconfig:")
        for k, v in d["config"].items():
            print(f"\t\t{k}: {v}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pattern", default="(.*).npy", help="re for matching npy files"
    )
    parser.add_argument(
        "--dirname", default="predictions", help="prediction data directory"
    )
    args = parser.parse_args()

    available = collect_available(args.pattern, args.dirname)

    while True:
        opt = input("enter dist|summary|3d|info: ")

        if opt == "dist":
            choices_menu(available, plot_dist)
        elif opt == "3d":
            choices_menu(available, plot_3d_menu)
        elif opt == "summary":
            plot_summary_menu(available)
        elif opt == "info":
            choices_menu(available, print_metadata)
        else:
            break
