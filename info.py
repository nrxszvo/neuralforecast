from functools import partial
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["keymap.back"].remove("left")
plt.rcParams["keymap.forward"].remove("right")
plt.rcParams["keymap.save"].remove("s")
plt.rcParams["keymap.pan"].remove("p")
plt.rcParams["keymap.zoom"].remove("o")
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.home"].remove("r")


def load(fn):
    return np.load(fn, allow_pickle=True).item()


def get_ys(d):
    yt = d["y_true"]
    yh = d["y_hat"]
    return yt, yh


def calc_mae(yt, yh, ax=(0, 1, 2, 3)):
    mae = np.abs(yt - yh)
    if len(ax) > 0:
        mae = mae.mean(axis=ax)
    return mae


def calc_smape(yt, yh, ax=(0, 1, 2, 3)):
    t = np.prod([yt.shape[i] for i in ax])
    smape = (200 / t) * np.sum(np.abs(yt - yh) / (np.abs(yt) + np.abs(yh)), axis=ax)
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
            smoothmag[i, j] = np.convolve(magdy[i, j], smoother, mode="valid") / nsmooth
    speed = np.min(smoothmag, axis=2)
    idx = np.argmin(smoothmag, axis=2)
    indices = (
        np.repeat(idx[:, :, None], nsmooth, axis=2) + np.arange(nsmooth)[None, None, :]
    )
    invspeed = -(speed - speed.max())
    while indices.ndim > ndim - 1:
        indices = indices[0]
    return invspeed, indices


def _update_data(smape_plt, mae_plt, speed_plt, smape_err, mae_err, invspeed, series):
    smape_plt.set_ydata(smape_err[series])
    mae_plt.set_ydata(mae_err[series])
    speed_plt.set_ydata(invspeed[series])


def onkeypress(fn, fig, ax, state, e):
    if e.key in ["left", "right"]:
        series = state["series"]
        n_series = state["nseries"]
        if e.key == "left":
            series = (n_series + series - 1) % n_series
        else:
            series = (series + 1) % n_series
        state["series"] = series
        fn(series)

        ax.set_title(f'{state["name"]} - Series {state["series"]}', fontsize=10)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    elif e.key in ["i", "m", "s"]:
        if e.key == "i":
            line = state["speed_plt"]
        elif e.key == "m":
            line = state["mae_plt"]
            line_lim = state["mae_lim"]
            other = state["smape_plt"]
            other_lim = state["smape_lim"]
        else:
            line = state["smape_plt"]
            line_lim = state["smape_lim"]
            other = state["mae_plt"]
            other_lim = state["mae_lim"]

        line.set_visible(not line.get_visible())

        if e.key in ["m", "s"]:
            if line.get_visible() and other.get_visible():
                lmin, lmax = state["global_lim"]
            elif line.get_visible():
                lmin, lmax = line_lim
            else:
                lmin, lmax = other_lim
            ax.set_ylim(lmin, lmax)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()


def _plot_errors_by_window(d, name, fig, ax, smape_win=None):
    yt, yh = get_ys(d)
    nseries, nwin, _, _ = yt.shape
    xax = np.arange(nwin)

    if smape_win == None:
        smape_win = calc_smape(yt, yh, ax=(2, 3))
    mae_win = calc_mae(yt, yh, ax=(2, 3))
    invspeed = calc_speed(yt)[0]

    state = {"series": 0, "nseries": nseries, "name": name}

    fig.suptitle("average error by window")

    ls = ""
    m = "o"
    ms = 1
    smape_plt = ax.plot(
        xax,
        smape_win[state["series"]],
        label="sMAPE",
        linestyle=ls,
        marker=m,
        markersize=ms,
    )[0]
    mae_plt = ax.plot(
        xax,
        mae_win[state["series"]],
        # label="mae",
        color="green",
        linestyle=ls,
        marker=m,
        markersize=ms,
    )[0]

    axt = ax.twinx()
    speed_plt = axt.plot(
        xax,
        invspeed[state["series"]],
        alpha=0.6,
        label="inverse speed",
        color="red",
        linestyle=ls,
        marker=m,
        markersize=ms,
    )[0]

    state["smape_plt"] = smape_plt
    state["mae_plt"] = mae_plt
    state["speed_plt"] = speed_plt

    state["smape_lim"] = (0.9 * smape_win.min(), 1.1 * smape_win.max())
    state["mae_lim"] = (0.9 * mae_win.min(), 1.1 * mae_win.max())
    state["global_lim"] = (
        min(state["smape_lim"][0], state["mae_lim"][0]),
        max(state["smape_lim"][1], state["mae_lim"][1]),
    )

    ax.set_ylim(*state["global_lim"])
    axt.set_ylim(0.9 * invspeed.min(), 1.1 * invspeed.max())

    update_fn = partial(
        _update_data,
        smape_plt,
        mae_plt,
        speed_plt,
        smape_win,
        mae_win,
        invspeed,
    )
    ax.set_ylabel("error")
    axt.set_ylabel("inverse speed")
    ax.legend([smape_plt, speed_plt], [smape_plt.get_label(), speed_plt.get_label()])
    ax.set_title(f'{state["name"]} - Series {state["series"]}', fontsize=10)

    ax.zorder = 1
    ax.patch.set_visible(False)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
    )

    def onclick(fig, ax, nwin, annot, d, name, state, e):
        if e.inaxes is ax:
            if e.dblclick:
                sidx = state["series"]
                widx = int(e.xdata + 0.5)
                annot.xy = (e.xdata, e.ydata)
                annot.set_text(f"{sidx}.{widx}")
                annot.set_visible(True)
                fig.canvas.draw_idle()
                plot_3d(d, name, sidx, widx)
                plt.show()
            else:
                annot.set_text("")
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "button_press_event", partial(onclick, fig, ax, nwin, annot, d, name, state)
    )

    fig.canvas.mpl_connect(
        "key_press_event", partial(onkeypress, update_fn, fig, ax, state)
    )

    return smape_win


def plot_dist(ds):
    axHist = plt.figure().add_subplot()
    axHistT = axHist.twinx()
    axHIdx = plt.figure().add_subplot()
    figWErr, axesWErr = plt.subplots(len(ds), 1)
    if len(ds) == 1:
        axesWErr = [axesWErr]

    # axes are: series, window, widx, dim
    for (d, name), axWErr in zip(ds, axesWErr):
        smape_win = _plot_errors_by_window(d, name, figWErr, axWErr)
        if name == ds[-1][1]:
            axWErr.set_xlabel("window #")

        yt, yh = get_ys(d)

        err_hidx = calc_smape(yt, yh, ax=(0, 1, 3))
        axHIdx.scatter(np.arange(len(err_hidx)), err_hidx, s=0.5, label=name)

        axHist.hist(smape_win.reshape(-1), bins=100, alpha=0.6)
        axHistT.hist(
            smape_win.reshape(-1),
            bins=100,
            cumulative=True,
            histtype="step",
            density=True,
            label="CDF",
            color="orange",
        )
    axHIdx.set_xlabel("horizon index")
    axHIdx.set_ylabel("sMAPE")
    axHIdx.legend()
    axHIdx.set_title("Average error by horizon index")

    axHist.set_xlabel("sMAPE")
    axHist.set_ylabel("# windows")
    axHist.set_yscale("log")
    axHistT.set_ylabel("cumulative likelihood")
    if len(ds) > 1:
        axHist.legend()
        axHist.set_title("Error distribution")
    else:
        axHist.set_title(ds[0][1] + " -- error distribution")
    plt.show()
    plt.close()


def plot_summary(ds):
    names = []
    errs = []
    for d, name in ds:
        names.append(name)
        yt, yh = get_ys(d)
        errs.append(calc_smape(yt, yh))
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


def plot_3d(d, name, sidx, widx):
    yt, yh = get_ys(d)
    ytw = yt[sidx, widx]
    yhw = yh[sidx, widx]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    _, highlight_idx = calc_speed(ytw)
    smape = calc_smape(ytw, yhw, ax=(0, 1))
    mae = calc_mae(ytw, yhw, ax=(0, 1))
    yt3d = ax.plot(*ytw.T, label="reference")[0]
    yh3d = ax.plot(*yhw.T, label="prediction", alpha=0.6)[0]
    ys3d = ax.plot(
        *ytw[highlight_idx].T,
        alpha=0.6,
        color="red",
        label="minimum speed region",
    )[0]
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)

    ax.set_title(f"Series {sidx}, Window {widx}, sMAPE Error: {smape:.2f}", loc="left")

    def concat_yt(yt, widx, nwinyt):
        inc = yt.shape[1] - 1
        start = widx - (inc * (nwinyt - 1))
        end = widx + inc
        return np.concatenate([yt[wi] for wi in np.arange(start, end, inc)])

    def update(yt, yh, ax, state, frame):
        widx = state["widx"] + frame
        sidx = state["sidx"]

        ytw = yt[sidx, widx]
        _, highlight_idx = calc_speed(ytw)
        smape = calc_smape(ytw, yh[sidx, widx], ax=(0, 1))
        state["yt3d"].set_data_3d(*ytw.T)
        state["yh3d"].set_data_3d(*yh[sidx, widx].T)
        state["ys3d"].set_data_3d(*ytw[highlight_idx[0]].T)
        ax.set_title(f"window {widx} - sMAPE Error: {smape:.2f}", loc="left")
        return state["yt3d"], state["yh3d"], state["ys3d"]

    def onkeypress(yt, yh, fig, ax, state, e):
        sidx = state["sidx"]
        widx = state["widx"]
        nseries, nwin, winsize, ndim = yt.shape
        if e.key == "i":
            viz = state["ys3d"].get_visible()
            label = ""
            if "ys3dlabel" in state:
                label = state["ys3dlabel"]
            state["ys3d"].set_visible(not viz)
            state["ys3dlabel"] = state["ys3d"].get_label()
            state["ys3d"].set_label(label)
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1, 1),
                bbox_transform=fig.transFigure,
            )
            fig.canvas.draw_idle()
        elif e.key == "b":
            if widx >= winsize - 1:
                state["nwinyt"] += 1
                ytw = concat_yt(yt[sidx], widx, state["nwinyt"])
                state["yt3d"].set_data_3d(*ytw.T)
                fig.canvas.draw_idle()
        elif e.key in ["left", "right"]:
            if e.key == "left":
                if widx > 0:
                    widx -= 1
                else:
                    sidx = (nseries + sidx - 1) % nseries
                    widx = nwin - 1
            elif e.key == "right":
                if widx < nwin - 1:
                    widx += 1
                else:
                    sidx = (sidx + 1) % nseries
                    widx = 0

            ytw = concat_yt(yt[sidx], widx, state["nwinyt"])
            _, highlight_idx = calc_speed(ytw)
            smape = calc_smape(ytw, yh[sidx, widx], ax=(0, 1))
            state["yt3d"].set_data_3d(*ytw.T)
            state["yh3d"].set_data_3d(*yh[sidx, widx].T)
            state["ys3d"].set_data_3d(*ytw[highlight_idx[0]].T)
            mae = state["mae"]
            state["widx"] = widx
            state["sidx"] = sidx
            # ax.set_title(f"Series {sidx}, Window {widx}, sMAPE Error: {smape:.2f}")
            ax.set_title(f"Window {widx} - sMAPE Error: {smape:.2f}", loc="left")

            fig.canvas.draw_idle()
        elif e.key == "r":
            state["ys3d"].set_label("point of departure")
            ax.legend(
                loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure
            )
            state["ys3d"].set_marker("o")
            ani = animation.FuncAnimation(
                fig,
                func=partial(
                    update,
                    yt,
                    yh,
                    ax,
                    {
                        "sidx": sidx,
                        "widx": widx,
                        "yt3d": state["yt3d"],
                        "yh3d": state["yh3d"],
                        "ys3d": state["ys3d"],
                    },
                ),
                frames=90,
                interval=100,
            )
            ani.save("animate.gif", dpi=50, writer=animation.PillowWriter(fps=10))

    fig.canvas.mpl_connect(
        "key_press_event",
        partial(
            onkeypress,
            yt,
            yh,
            fig,
            ax,
            {
                "sidx": sidx,
                "widx": widx,
                "yt3d": yt3d,
                "yh3d": yh3d,
                "ys3d": ys3d,
                "smape": smape,
                "mae": mae,
                "nwinyt": 1,
            },
        ),
    )


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
            available.append((f"{dirname}/{fn}", m.group(1)))
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
                fn([(load(fn), name) for fn, name in choices])
                choices = []
            else:
                break
        else:
            try:
                c = int(inp)
                name = available[c][1]
                n = input("optional title for plot: ")
                if n != "":
                    name = n
                choices.append((available[c][0], name))
            except Exception as e:
                print(e)


def plot_3d_menu(ds):
    if len(ds) > 1:
        raise Exception("plot_3d only supports one dataset")
    d, name = ds[0]
    try:
        nser = d["y_true"].shape[0]
        nwin = d["y_true"].shape[1]
        resp = input(f"{name}: series and window to plot [0, {nser-1}].[0, {nwin-1}]: ")
        sidx, widx = resp.split(".")
        plot_3d(d, name, int(sidx), int(widx))
    except Exception as e:
        print(e)
    plt.show()
    plt.close()


def print_metadata(ds):
    for d, name in ds:
        yt, yh = get_ys(d)
        mae = calc_mae(yt, yh)
        smape = calc_smape(yt, yh)
        print()
        print(name)
        if "dataset" in d:
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
