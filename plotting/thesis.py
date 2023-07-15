import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
   df = pd.read_csv(sys.stdin,sep='\t', dtype=int)
   results = df[['padding', 'time taken']]
   baseline = df [['padding', 'time taken (baseline)']]
   baseline['time taken'] = baseline['time taken (baseline)']
   baseline = baseline.drop(columns=['time taken (baseline)'])
   results['name'] = "Experiment"
   baseline['name'] = "Baseline"
   # df = pd.concat([results, baseline])
   print(df)
   _ = sns.lineplot(x='padding', y='time taken', data=df)
   _ = sns.lineplot(x='padding', y='time taken (baseline)', data=df)
   plt.show()
   

if __name__ == "__main__":
    main()
