import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from functools import partial

plt.rcParams["keymap.fullscreen"].remove("f")
plt.rcParams["keymap.back"].remove("left")
plt.rcParams["keymap.forward"].remove("right")
plt.rcParams["keymap.save"].remove("s")
plt.rcParams["keymap.pan"].remove("p")
plt.rcParams["keymap.zoom"].remove("o")
plt.rcParams["keymap.quit"].remove("q")


def read_csv(csvfn, offset=0, nseries=10):
    ndim = 3
    slen = 10000
    with pd.read_csv(
        sys.argv[1], skiprows=offset, chunksize=ndim * slen * nseries
    ) as reader:
        df = reader.__iter__().get_chunk()
        sids = df.unique_id.unique()
        data = []
        for id in sids:
            d = df[df.unique_id == id].y.to_numpy()
            data.append(d.reshape(-1, 3))
        return np.stack(data)


def compare_series(ys):
    sidxs = np.arange(ys.shape[0])
    y3ds = []
    wstart = 0
    wend = 100
    inc = 100
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    for sidx in sidxs:
        yw = ys[sidx, wstart:wend]
        y3d = ax.plot(*yw.T, label=sidx)[0]
        y3ds.append(y3d)
    # ax.legend()
    # ax.set_title("Lorenz Attractor - Initial Conditions")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    def onkeypress(ys, fig, ax, state, e):
        sidxs = state["sidxs"]
        y3ds = state["y3ds"]
        wstart = state["wstart"]
        wend = state["wend"]
        inc = state["inc"]
        nseries, npoints, ndim = ys.shape
        if e.key in ["q", "w", "o", "p"]:
            idx = 0 if e.key in ["q", "w"] else 1
            if e.key in ["q", "o"]:
                sidxs[idx] = (sidxs[idx] + 1) % nseries
            else:
                sidxs[idx] = (nseries + sidxs[idx] - 1) % nseries
            for sidx, y3d in zip(sidxs, y3ds):
                yw = ys[sidx, wstart:wend]
                y3d.set_data_3d(*yw.T)
                y3d.set_label(sidx)
            ax.legend()
            fig.canvas.draw_idle()

        if e.key in ["f", "b"]:
            if e.key == "f":
                if wend + inc < npoints:
                    wend += inc
            else:
                if wend - wstart > inc:
                    wend -= inc
                elif wstart - inc >= 0:
                    wstart -= inc
            for sidx, y3d in zip(sidxs, y3ds):
                yw = ys[sidx, wstart:wend]
                y3d.set_data_3d(*yw.T)
                y3d.set_label(sidx)
            state["wstart"] = wstart
            state["wend"] = wend
            fig.canvas.draw_idle()
        elif e.key in ["left", "right"]:
            if e.key == "left":
                if wstart > 0:
                    wstart -= 1
                    wend -= 1
                else:
                    sidxs = [(nseries + sidx - 1) % nseries for sidx in sidxs]
                    wstart = npoints - inc
                    wend = npoints
            elif e.key == "right":
                if wend < npoints - 1:
                    wend += 1
                    wstart += 1
                else:
                    sidxs = [(sidx + 1) % nseries for sidx in sidxs]
                    wstart = 0
                    wend = inc

            for sidx, y3d in zip(sidxs, y3ds):
                yw = ys[sidx, wstart:wend]
                y3d.set_data_3d(*yw.T)
                y3d.set_label(sidx)
            state["wstart"] = wstart
            state["wend"] = wend
            state["sidxs"] = sidxs
            ax.legend()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "key_press_event",
        partial(
            onkeypress,
            ys,
            fig,
            ax,
            {
                "sidxs": sidxs,
                "wstart": wstart,
                "wend": wend,
                "y3ds": y3ds,
                "inc": inc,
            },
        ),
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    import sys

    ys = read_csv(sys.argv[1])
    compare_series(ys)
