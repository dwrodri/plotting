import sys
from typing import List, Tuple
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
BRANCH_LOG_LIMIT = 2_000_000


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
                    bool(int(is_br)),  # Is this instruction a branch?
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


def load_decode_log_precomputed(filename: str) -> pd.DataFrame:
    return pd.read_parquet(filename)


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


def find_speculated_instructions(
    wb_df: pd.DataFrame, decode_df: pd.DataFrame
) -> pd.DataFrame:
    # if we haven't done commit matching yet, then we need to do that
    if not decode_df["Has Matching Commit"].any():
        for _, wb_tsc, wb_pc in tqdm(wb_df.itertuples()):
            decode_df.iat[
                decode_df[
                    (0 < (wb_tsc - decode_df["Timestamp"]) < 500)
                    & (decode_df["PC"] == wb_pc)
                    & (~decode_df["Has Matching Commit"])
                ].first_valid_index(),
                2,
            ] = True
        decode_df.to_parquet("decoded_with_spec.parquet")
    decode_df.to_parquet("decoded_with_spec.parquet")
    return decode_df


def make_plot():
    branch_df = load_branch_log(
        "/Users/dwrodri/Codespace/plotting/data/small-gshare-branch.log"
    )
    branch_df["Timestamp"] = branch_df["Timestamp"].astype(int)
    branch_df["PC"] = branch_df["PC"].astype(int)
    max_tsc = branch_df["Timestamp"].max()
    taken = branch_df[branch_df["Is Branch"] & branch_df["Taken"]]
    not_taken = branch_df[branch_df["Is Branch"] & ~branch_df["Taken"]]
    jumps = branch_df[branch_df["Is Jump"]]
    # decode_df = load_decode_log("/Users/dwrodri/Codespace/plotting/data/decoded.txt")
    # decode_df = (
    #     decode_df[decode_df["Timestamp"] <= max_tsc]
    #     .sort_values(by="Timestamp", ascending=False)
    #     .reset_index(drop=True)
    # )
    # decode_df["Has Matching Commit"] = False
    decode_df = load_decode_log_precomputed("data/decoded_with_spec.parquet")
    speculated_df = decode_df[~decode_df["Has Matching Commit"]]
    writeback_df = load_commit_log(
        "/Users/dwrodri/Codespace/plotting/data/writebacked.txt"
    )
    writeback_df = (
        writeback_df[writeback_df["Timestamp"] <= max_tsc]
        .sort_values(by="Timestamp", ascending=False)
        .reset_index(drop=True)
    )
    writeback_df.to_parquet("writeback.parquet")
    print(len(writeback_df))
    print("got this far")
    sns.set(font_scale=1.60)
    ax: plt.Axes = plt.gca()
    ax.plot(decode_df["Timestamp"], decode_df["PC"], "k.", label="decoded µOP")
    ax.plot(writeback_df["Timestamp"], writeback_df["PC"], "m.", label="retired µOP")
    ax.plot(
        speculated_df["Timestamp"],
        speculated_df["PC"],
        "c*",
        label="speculated µOP",
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
    ax.set_title("Speculative Behavior during Spectre Attack")
    ax.set_xlabel("# of Cycles")
    ax.set_ylabel("Program Counter")
    ax.set_yticklabels(ax.get_yticks(), rotation=45)
    # ax.get_yaxis().set_ticks(df["PC"].unique().data.tolist())
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(float_to_addr))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    make_plot()
