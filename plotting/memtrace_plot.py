import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def main(memtrace_filename: str):
    rows = []
    with open(memtrace_filename, "r") as fp:
        for line in tqdm(fp, total=12231340):
            if "MT" == line[:2]:
                _, tsc, cmd, _, _, addr, *_ = line.split()
                rows.append([int(elem, 16) for elem in [tsc, cmd, addr]])
    df = pd.DataFrame(rows, columns=["Timestamp", "Command", "Address"], dtype=np.uint64)
    sns.scatterplot(x="Timestamp", y="Address", hue="Command", data=df)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
