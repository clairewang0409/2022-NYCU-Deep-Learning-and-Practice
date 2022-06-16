import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('./result/score_0425.csv')

plt.plot(df)
plt.xlabel('1k episodes')
plt.ylabel('mean score')
plt.savefig('result_0425.png')