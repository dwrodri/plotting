import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sns.set_theme(style="darkgrid", font_scale=1.75)


def load_data(filename: str) -> pd.DataFrame:
    logging.debug(f"Parsing {filename}")
    df = pd.read_csv(
        filename,
        sep=" ",
        names=["debug_fsrc", "taken", "is_branch", "is_jal", "is_jalr", "debug_pc"],
    )
    df["debug_pc"] = df["debug_pc"].apply(lambda n: int(n, 16))
    df[["taken", "is_branch", "is_jal", "is_jalr"]] = df[
        ["taken", "is_branch", "is_jal", "is_jalr"]
    ].astype(bool)
    df["branch_idx"] = df.index
    df["predictor"] = filename.split("-")[1]
    return df


dfs = [load_data(filename) for filename in sys.argv[1:]]
dfs = [df[df["is_branch"]] for df in dfs]
dfs = [df[["debug_pc", "branch_idx", "taken", "predictor"]] for df in dfs]
for df in dfs:
    df["branch_id"] = df["debug"]
# df = dfs[0].join(dfs[1], on="debug_pc", lsuffix="gshare", rsuffix="ltage")
# df = df.join(dfs[2], on="debug_pc", lsuffix="", rsuffix="tournament")
