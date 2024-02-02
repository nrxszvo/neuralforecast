import os
import numpy as np
import pandas as pd
import sys

sys.path.append("../dysts")
from dysts.base import make_trajectory_ensemble
import dysts.flows as flows
import json


def make_randic(name, nseq, seqlen, ic_perturb):
    fn = f"{name}_{nseq}x{seqlen}_ic-{ic_perturb}"
    if name == "Lorenz":
        model = flows.Lorenz()
    else:
        raise Exception
    perturb = 1 + (ic_perturb / 2) - ic_perturb * np.random.random((nseq, 3))
    model.ic = model.ic[None, :] * perturb
    tpts, sol = model.make_trajectory(n=seqlen, return_times=True)

    df = pd.DataFrame(columns=["unique_id", "ds", "y"]).astype(
        {"unique_id": str, "ds": np.int64, "y": np.float64}
    )

    md = {"ic": model.ic.tolist(), "dt": tpts[1]}
    with open(f"{fn}.json", "w") as f:
        json.dump(md, f, sort_keys=True, indent=4)

    for i in range(nseq):
        y = sol[i].reshape(-1)
        ds = np.arange(len(y))
        uid = [f"{name}-{i}"] * len(y)
        df = pd.concat(
            [df, pd.DataFrame.from_dict({"unique_id": uid, "ds": ds, "y": y})]
        )
    df.to_csv(f"{fn}.csv", index=False)


def make(seqlen=100, pts_per_period=75, resample=True):
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
    import sys

    # seqlen = int(sys.argv[1])
    # make(seqlen)
    make_randic("Lorenz", 25, 10000, 0.02)
