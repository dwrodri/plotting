import sys
from typing import List, Tuple
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

logging.basicConfig(level=logging.INFO)
BRANCH_LOG_LIMIT = 100_000


def print_passthru(elem):
    print(elem)
    return elem


def load_branch_log(filename: str, limit: int = BRANCH_LOG_LIMIT) -> pd.DataFrame:
    """
    Parse lines from the branch printf log dumped from a Boom simulation into a DataFrame
    """
    logging.debug(f"Loading branch log: {filename}")
    with open(filename) as fp:
        return pd.DataFrame.from_records(
            (
                (
                    int(tsc, 16),  # current cycle
                    bool(int(is_br)),  # Is this insutrction a branch?
                    bool(int(is_jal) | int(is_jalr)),  # Is this instruction a jump?
                    bool(int(taken)),  # Was this branch taken once resolved ?
                    int(pc, 16),  # PC of instruction
                )
                for line in fp
                if line[0].isdigit()
                for tsc, _, taken, is_br, is_jal, is_jalr, pc in [line.strip().split()]
                if int(pc, 16) >= 0x80001000
            ),
            columns=["Timestamp", "Is Branch", "Is Jump", "Taken", "PC"],
            nrows=limit,
        )


def load_decode_log(filename: str) -> pd.DataFrame:
    """
    Parse lines from out custom decode log into a DataFrame
    """
    logging.info(f"Loading decode log: {filename}")
    with open(filename) as fp:
        return pd.DataFrame.from_records(
            np.fromiter(
                (
                    (int(tsc, 16), int(pc, 16))
                    for line in fp
                    for tsc, pc in [line.strip().split(",")]
                    if int(pc, 16) >= 0x80001000
                ),
                dtype=np.dtype((np.int_, 2)),
            ),
            columns=["Timestamp", "PC"],
            nrows=2_005_542,
        )


def load_commit_log(filename: str) -> pd.DataFrame:
    """
    Parse lines from out custom writeback log into a DataFrame
    """
    logging.info(f"Loading commit log: {filename}")
    with open(filename) as fp:
        return pd.DataFrame.from_records(
            np.fromiter(
                (
                    (int(tsc, 16), int(pc, 16))
                    for line in fp
                    for tsc, _, pc, *_ in [line.strip().split(",")]
                    if int(pc, 16) >= 0x80001000
                ),
                dtype=np.dtype((np.int_, 2)),
            ),
            columns=["Timestamp", "PC"],
        )


def float_to_addr(x, pos) -> str:
    """
    Helper function for printing hex values on matplotlib axes
    """
    return hex(int(x))


def make_plot():
    branch_df = load_branch_log(
        "/Users/dwrodri/Codespace/plotting/data/small-tagel-branch.log"
    )
    branch_df["Timestamp"] = branch_df["Timestamp"].astype(int)
    branch_df["PC"] = branch_df["PC"].astype(int)
    max_tsc = branch_df["Timestamp"].max()
    taken = branch_df[branch_df["Is Branch"] & branch_df["Taken"]]
    not_taken = branch_df[branch_df["Is Branch"] & ~branch_df["Taken"]]
    jumps = branch_df[branch_df["Is Jump"]]
    decode_df = load_decode_log(
        "/Users/dwrodri/Codespace/plotting/data/tagel-decoded.txt"
    )
    decode_df = (
        decode_df[decode_df["Timestamp"] <= max_tsc]
        .sort_values(by="Timestamp")
        .reset_index(drop=True)
    )
    writeback_df = load_commit_log(
        "/Users/dwrodri/Codespace/plotting/data/tagel-writebacked.txt"
    )
    writeback_df = (
        writeback_df[writeback_df["Timestamp"] <= max_tsc]
        .sort_values(by="Timestamp")
        .reset_index(drop=True)
    )
    print(len(writeback_df))
    print("got this far")
    unspeculated_df = pd.merge(left=writeback_df, right=decode_df, on="PC", how="right")
    # speculated_df = decode_df[unspeculated_df["_merge"] == "left_only"]
    print(unspeculated_df.tail())
    print(decode_df.head())
    print(writeback_df.head())
    sys.exit()
    sns.set(font_scale=1.60)
    ax: plt.Axes = plt.gca()
    ax.plot(decode_df["Timestamp"], decode_df["PC"], "k.", label="decoded µOP")
    ax.plot(writeback_df["Timestamp"], writeback_df["PC"], "m.", label="retired µOP")
    ax.plot(
        unspeculated_df["Timestamp"],
        unspeculated_df["PC"],
        "co",
        label="unspeculated µOP",
    )
    ax.plot(
        taken["Timestamp"],
        taken["PC"],
        "ro",
        label="Br. Taken",
    )
    ax.plot(
        not_taken["Timestamp"],
        not_taken["PC"],
        "bo",
        label="Not Taken",
    )
    ax.plot(jumps["Timestamp"], jumps["PC"], "go", label="Jump")
    ax.set_title("TAGE-L Predictor response to Spectre v1 attack")
    ax.set_xlabel("# of Cycles")
    ax.set_ylabel("Program Counter")
    # ax.get_yaxis().set_ticks(df["PC"].unique().data.tolist())
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(float_to_addr))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    make_plot()
