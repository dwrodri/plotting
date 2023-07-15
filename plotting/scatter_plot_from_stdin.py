import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(font_scale=1.75)

data = [int(line.strip()) for line in sys.stdin.readlines()]
plt.plot(np.diff(data), ".")
plt.yscale('log')
plt.show()
