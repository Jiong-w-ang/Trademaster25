import pandas as pd
import numpy as np

a = pd.read_csv("../data/train.csv", usecols=["id","target_short","target_medium","target_long"]).sort_values("id")
b = pd.read_csv("../data/train_v2.csv", usecols=["id","target_short","target_medium","target_long"]).sort_values("id")

diff = a[["target_short","target_medium","target_long"]].values - b[["target_short","target_medium","target_long"]].values

print("max abs diff:", np.max(np.abs(diff)))
print("mean abs diff (short,med,long):", np.mean(np.abs(diff), axis=0))
print("corr (short,med,long):",
      np.corrcoef(a["target_short"], b["target_short"])[0,1],
      np.corrcoef(a["target_medium"], b["target_medium"])[0,1],
      np.corrcoef(a["target_long"], b["target_long"])[0,1])
