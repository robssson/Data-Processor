import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import hsv
import matplotlib.patches as mpatches
from pmdarima.arima import auto_arima, ADFTest
import os


processed_df = pd.read_csv('data/input/indexProcessed.csv')
data_df = pd.read_csv('data/input/indexData.csv')
info_df = pd.read_csv('data/input/indexInfo.csv')

print(processed_df.describe())
processed_df = processed_df.merge(info_df, on='Index')
print(processed_df)
print(processed_df.isnull().any())
print(print(processed_df.dtypes))
processed_df['Date'] = pd.to_datetime(processed_df['Date'])
processed_df['Index'] = processed_df['Index'].astype('category')
processed_df['Region'] = processed_df['Region'].astype('category')
processed_df['Exchange'] = processed_df['Exchange'].astype('category')
processed_df['Currency'] = processed_df['Currency'].astype('category')

stocks = pd.unique(processed_df['Index'])
stock_dfs = []
for stock in stocks:
    stock_dfs.append(processed_df[processed_df['Index'] == stock])
print(stock_dfs[2])

fig, ax = plt.subplots(figsize=(20, 15))
patches = []
for i, stock_df in enumerate(stock_dfs):
    color = (hsv(i/len(stock_dfs)))
    sns.lineplot(ax=ax, x=stock_df['Date'], y=stock_df['CloseUSD'], color=color)
    patches.append(mpatches.Patch(color=color, label=stock_df['Index'].iloc[0]))
ax.legend(handles=patches)

stock_dfs = {stock_df['Index'].iloc[0]: stock_df for stock_df in stock_dfs}  # Convert list to dict
sns.lineplot(ax=ax, data=processed_df, x='Date', y='CloseUSD')
fig.show()

NSEI = stock_dfs['NSEI']
NSEI = NSEI.sort_values('Date').reset_index(drop=True)
train_df = NSEI[NSEI['Date'] < '2018'][['CloseUSD', 'Date']].set_index('Date')
test_df = NSEI[NSEI['Date'] > '2018'][['CloseUSD', 'Date']].set_index('Date')
plt.plot(train_df)
plt.plot(test_df)
plt.legend(['train', 'test'])
plt.show()

NSEI = stock_dfs['NSEI']
NSEI = NSEI.sort_values('Date').reset_index(drop=True)
train_df = NSEI[NSEI['Date'] < '2018'][['CloseUSD', 'Date']].set_index('Date')
test_df = NSEI[NSEI['Date'] > '2018'][['CloseUSD', 'Date']].set_index('Date')
plt.plot(train_df)
plt.plot(test_df)
plt.legend(['train', 'test'])
plt.show()

arima_model = auto_arima(train_df)
arima_model.summary()
prediction = pd.DataFrame(arima_model.predict(n_periods=len(test_df)), index=test_df.index)
prediction.columns = ['CloseUSD']
print(prediction)
plt.figure(figsize=(10,10))
sns.lineplot(x=train_df.index, y=train_df['CloseUSD'])
sns.lineplot(x=test_df.index, y=test_df['CloseUSD'])
sns.lineplot(x=prediction.index, y=prediction['CloseUSD'])
plt.legend(['train', 'test', 'prediction'])
plt.show()
