import os
import numpy as np
import pandas as pd
from dysts.base import make_trajectory_ensemble


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
