import functools
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import TypeVar


T = TypeVar("T")


def print_passthru(elem: T) -> T:
    print(f"TYPE: {type(elem)} VAL: {elem}")
    return elem


def main(filename: str):
    with open(filename) as fp:
        return np.fromiter(
            tqdm(
                (int(a, 16), int(b, 16))
                for a, b in (line.strip().split(",") for line in fp)
            ),
            dtype=np.dtype((int, 2)),
        )


if __name__ == "__main__":
    data = main(sys.argv[1])
