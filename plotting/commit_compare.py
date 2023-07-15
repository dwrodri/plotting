import pathlib
import sys
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5)


def parse_trace(lines: List[str]) -> List[int]:
    result = []
    for line in lines:
        elems = line.split()
        try:
            _, pc, *_ = elems
            if "0x000000008" in pc:
                result.append(int(pc[2:], 16))
        except:
            print(line)
            sys.exit(-1)
    return result


def main(trace_files: List[str]):
    for trace in trace_files:
        with open(trace, "r") as fp:
            plt.plot(parse_trace(fp.readlines()), ".", label=pathlib.Path(trace).stem)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
