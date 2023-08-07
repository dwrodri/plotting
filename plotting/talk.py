import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
oh_seven = [0.275, 0.411, 0.220, 0.055, 0.11, 0.03]
twenty_twelve = [0.35, 0.48, 0.13, 0.018, 0.022]
twenty_fourteen = [0.32, 0.45, 0.17, 0.05, 0.005]

plt.plot(range(1, 8), oh_seven, "-", label=2007)
plt.plot(range(1, 8), twenty_twelve, "-", label=2012)
plt.plot(range(1, 8), twenty_fourteen, "-", label=2014)
plt.title("Number of spins required to produce a valid chip")
plt.xlabel("")
plt.legend(title="Year")
