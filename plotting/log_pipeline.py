import numpy as np
import pandas as pd
import logging
import pathlib
import sys

logging.basicConfig(level=logging.INFO)


def load_branch_log(filename: str) -> pd.DataFrame:
    """
    Parse lines from the branch printf log dumped from a Boom simulation into a DataFrame
    """
    logging.debug(f"Loading branch log: {filename}")
    checkpoint_path = pathlib.Path(
        filename[: filename.index(".", filename.index("."))] + "_branch.parquet"
    )
    if checkpoint_path.exists():
        return pd.read_parquet(checkpoint_path)
    else:
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
                    for tsc, _, taken, is_br, is_jal, is_jalr, pc in [
                        line.strip().split()
                    ]
                    if int(pc, 16) >= 0x80001000
                ),
                columns=["Timestamp", "Is Branch", "Is Jump", "Taken", "PC"],
            )
            return df


def load_decode_log(filename: str) -> pd.DataFrame:
    checkpoint_path = pathlib.Path(
        filename[: filename.index(".", filename.index("."))] + "_decode.parquet"
    )
    with open(filename) as fp:
        df = pd.DataFrame.from_records(
            np.fromiter(
                (
                    (int(tsc, 16), int(pc, 16))
                    for line in fp
                    for _, tsc, pc in [line.strip().split(",")]
                    if "DECODE" in line
                    if int(pc, 16) >= 0x80001000
                ),
                dtype=np.dtype((np.int_, 2)),
            ),
            columns=["Timestamp", "PC"],
        )
        checkpoint_path = pathlib.Path(
            filename[: filename.index(".")] + "_decode.parquet"
        )
        if not checkpoint_path.exists():
            df.to_parquet(str(checkpoint_path))
        return df


def load_writeback_log(filename: str) -> pd.DataFrame:
    checkpoint_path = pathlib.Path(
        filename[: filename.index(".", filename.index("."))] + "_writeback.parquet"
    )
    with open(filename) as fp:
        df = pd.DataFrame.from_records(
            np.fromiter(
                (
                    (int(tsc, 16), int(pc, 16))
                    for line in fp
                    for _, tsc, pc in [line.strip().split(",")]
                    if "DECODE" in line
                    if int(pc, 16) >= 0x80001000
                ),
                dtype=np.dtype((np.int_, 2)),
            ),
            columns=["Timestamp", "PC"],
        )
        checkpoint_path = pathlib.Path(
            filename[: filename.index(".")] + "_decode.parquet"
        )
        if not checkpoint_path.exists():
            df.to_parquet(str(checkpoint_path))
        return df


def update_decode_log_with_speculations(
    wb_df: pd.DataFrame,
    decode_df: pd.DataFrame,
    filename: pathlib.Path = pathlib.Path("decoded_with_spec.parquet"),
) -> pd.DataFrame:
    # if we haven't done commit matching yet, then we need to do that
    if "Has Matching Commit" not in decode_df.columns:
        decode_df["Has Matching Commit"] = False
        for _, wb_tsc, wb_pc in wb_df.itertuples():
            decode_df.iat[
                decode_df[
                    (decode_df["PC"] == wb_pc)
                    & (decode_df["Timestamp"] <= wb_tsc)
                    & (~decode_df["Has Matching Commit"])
                ].first_valid_index(),
                2,
            ] = True
    decode_df.to_parquet(filename)
    return decode_df


if __name__ == "__main__":
    filename = sys.argv[1]
    if "commit" in filename:
        writeback_df = load_writeback_log(filename)
        decode_df = load_decode_log(filename)
        checkpoint_path = pathlib.Path(
            filename[: filename.index(".", filename.index("."))]
            + "_speculated_decode.parquet"
        )
        _ = update_decode_log_with_speculations(
            writeback_df, decode_df, checkpoint_path
        )
