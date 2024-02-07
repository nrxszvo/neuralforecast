import os
import numpy as np
import pandas as pd
import sys

sys.path.append("../dysts")
from dysts.base import make_trajectory_ensemble
import dysts.flows as flows
import json


def make_multi_ic(name, n_ic, seqlen, ic_perturb, fn=None):
    if fn is None:
        fn = f"{name}_{n_ic}x{seqlen}_ic-{ic_perturb}"
    model = getattr(flows, name)()
    model_md = model._load_data()
    ndim = model_md["embedding_dimension"]
    perturb = 1 + ic_perturb - (2 * ic_perturb) * np.random.random((n_ic, ndim))
    perturb[0] = np.ones(ndim)  # first ic are defaults
    model.ic = model.ic[None, :] * perturb
    tpts, sol = model.make_trajectory(n=seqlen, return_times=True)

    df = pd.DataFrame(columns=["unique_id", "ds", "y"]).astype(
        {"unique_id": str, "ds": np.int64, "y": np.float64}
    )

    md = {
        "model": name,
        "ndim": ndim,
        "dt": tpts[1],
        "lyapunov_time": 1 / model_md["maximum_lyapunov_estimated"],
        "period": model_md["period"],
        "ic": model.ic.tolist(),
    }
    with open(f"{fn}.json", "w") as f:
        json.dump(md, f, sort_keys=True, indent=4)

    for i in range(n_ic):
        y = sol[i].reshape(-1)
        ds = np.arange(len(y))
        uid = [f"{name}-{i}"] * len(y)
        df = pd.concat(
            [df, pd.DataFrame.from_dict({"unique_id": uid, "ds": ds, "y": y})]
        )
    df.to_csv(f"{fn}.csv", index=False)


def make_ensemble(seqlen=100, pts_per_period=75, resample=True):
    csvfn = f"dysts_ensemble_3d-{seqlen}.csv"
    if os.path.exists(csvfn):
        print(f"reading from {csvfn}")
        df = pd.read_csv(csvfn)
    else:
        print(f"generating {csvfn}...", end=" ", flush=True)
        ens = make_trajectory_ensemble(
            seqlen, resample=resample, pts_per_period=pts_per_period
        )
        df = pd.DataFrame(columns=["unique_id", "ds", "y"]).astype(
            {"unique_id": str, "ds": np.int64, "y": np.float64}
        )
        for k in ens:
            if ens[k].shape[1] == 3:
                y = ens[k].reshape(-1)
                ds = np.arange(len(y))
                uid = [k] * len(y)
                df = pd.concat(
                    [df, pd.DataFrame.from_dict({"unique_id": uid, "ds": ds, "y": y})]
                )
        df.to_csv(csvfn, index=False)
        print("done")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", default="Lorenz", help="model name", choices=["Lorenz"]
    )
    parser.add_argument("--seqlen", default=10000, help="sequence length", type=int)
    parser.add_argument(
        "--ic_perturb",
        default=0.01,
        help="percent random perturbation of initial conditions",
        type=float,
    )
    parser.add_argument(
        "--num_ic",
        default=100,
        help="number of unique initial conditions to generate",
        type=int,
    )
    parser.add_argument("--fn", default=None, help="output filename")
    args = parser.parse_args()
    make_multi_ic(args.model, args.num_ic, args.seqlen, args.ic_perturb, args.fn)
