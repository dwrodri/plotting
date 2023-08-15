import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

BRANCH_LIMIT = 2_000_000


def inline_print(val):
    print(val)
    return val


def load_branch_log(filename: pathlib.Path) -> pd.DataFrame:
    """
    Parse lines from the branch printf log dumped from a Boom simulation into a DataFrame
    """
    logging.debug(f"Loading branch log: {filename}")
    checkpoint_path = filename.with_stem(filename.stem + "_branch").with_suffix(
        ".parquet"
    )
    with open(filename) as fp:
        df = pd.DataFrame.from_records(
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
            # nrows=BRANCH_LIMIT,
        )
        df.to_parquet(filename.with_suffix(".parquet"))
        return df


def load_decode_log(filename: pathlib.Path) -> pd.DataFrame:
    checkpoint_path = filename.with_stem(filename.stem + "_decode").with_suffix(
        ".parquet"
    )
    with open(filename) as fp:
        df = pd.DataFrame.from_records(
            np.fromiter(
                (
                    (int(tsc, 16), int(pc, 16))
                    for _, tsc, pc in (
                        line.strip().split() for line in fp if "DECODE" in line
                    )
                    if int(pc, 16) >= 0x80001000
                ),
                dtype=np.dtype((np.int_, 2)),
            ),
            columns=["Timestamp", "PC"],
        )
        df.to_parquet(checkpoint_path)
        return df


def load_writeback_log(filename: pathlib.Path) -> pd.DataFrame:
    checkpoint_path = filename.with_stem(filename.stem + "_writeback").with_suffix(
        ".parquet"
    )
    with open(filename) as fp:
        df = pd.DataFrame.from_records(
            np.fromiter(
                (
                    (int(tsc, 16), int(pc, 16))
                    for tsc, _, pc, *_ in (
                        line.strip().split() for line in fp if "0x" == line[:2]
                    )
                    if int(pc, 16) >= 0x80001000
                ),
                dtype=np.dtype((np.int_, 2)),
            ),
            columns=["Timestamp", "PC"],
        )
        df.to_parquet(checkpoint_path)
        return df


def update_decode_log_with_speculations(
    wb_df: pd.DataFrame,
    decode_df: pd.DataFrame,
    filename: pathlib.Path = pathlib.Path("decoded_with_spec.parquet"),
) -> pd.DataFrame:
    wb_df["WB Timestamp"] = wb_df["Timestamp"]
    decode_df["Decode Timestamp"] = decode_df["Timestamp"]
    # merge asof is a fast approximate inner join
    merged = pd.merge_asof(
        decode_df,
        wb_df,
        left_on="Decode Timestamp",
        right_on="WB Timestamp",
        by="PC",
        direction="forward",
        allow_exact_matches=False,
        suffixes=["_decode", "_wb"],
    )
    # groupby the associated wb timestamp and then only return the max to get most likely decode time
    not_speculated = merged.groupby("Timestamp_wb").agg(max).reset_index(drop=True)["Decode Timestamp"]
    # we care about decodes that DONT have an associated writeback
    decode_df["Speculated"] = ~decode_df["Timestamp"].isin(not_speculated)
    decode_df.to_parquet(filename)
    return decode_df


if __name__ == "__main__":
    args = sys.argv[1].split()
    branch_filename, commit_filename = map(pathlib.Path, args)
    branch_df = load_branch_log(branch_filename)
    max_tsc = branch_df["Timestamp"].max()
    writeback_df = load_writeback_log(commit_filename)
    decode_df = load_decode_log(commit_filename)
    writeback_df = (
        writeback_df[writeback_df["Timestamp"] <= max_tsc]
        .sort_values(by="Timestamp")
        .reset_index(drop=True)
    )
    decode_df = (
        decode_df[decode_df["Timestamp"] <= max_tsc]
        .sort_values(by="Timestamp")
        .reset_index(drop=True)
    )
    checkpoint_path = commit_filename.with_stem(
        commit_filename.stem + "_speculated"
    ).with_suffix(".parquet")
    _ = update_decode_log_with_speculations(writeback_df, decode_df, checkpoint_path)
