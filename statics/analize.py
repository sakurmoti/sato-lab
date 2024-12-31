import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=["x", "y", "seq"])

st = set()

with open("../slurm/logs/GA/506_RNA.out") as f:
    lines = f.readlines()
    for line in lines:
        if "seq" in line:
            parsed = line.split()
            x = float(parsed[0])
            y = float(parsed[1])
            seq = parsed[-1]

            if seq in st:
                continue

            df.loc[len(df)] = [x, y, seq]
            st.add(seq)

print(df)

plt.scatter(df["x"], df["y"])
plt.xlabel("TE")
plt.ylabel("Deg")
plt.savefig("analize.png")


