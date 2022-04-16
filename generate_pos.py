import numpy as np
import pandas as pd
import pickle

threshold = 0.004

with open("y_predicted.pydata","rb") as f:
    df = pickle.load(f)

df['pred_pct_change'] = df['y_predicted'].pct_change().shift(-1)
df['true_pct_change'] = df['y_predicted'].pct_change().shift(-1)

df = df.iloc[:-1,:]

df['action'] = [1 if df.iloc[i,3] > threshold else -1 if df.iloc[i,3] < -threshold else 0 for i in range(len(df))]
df = df.where(df['action']!=0, inplace=False).dropna(how='any')

print(df)
'''
df['pos'] = None
df.iloc[0,4] = 0
for i in range(1,len(df)):
    df.iloc[i,4] = df.iloc[i-1,4] + df.iloc[i,3] '''

'''# return of this actoin 
df['return'] = (df['true_pct_change'] * df['action'] - threshold) 
df['cum_return'] = df['return'].cumsum()
print("The expected return is: ",df.iloc[-1,-1] )
'''

# generate price_path data to be feed into backtest bot 
price_df = pd.DataFrame({'Timeframe': df['timestamp'],
                            'high': df['y_true'],
                            'low': df['y_true'] ,
                            'open': df['y_true'],
                            'close': df['y_true'],
                            'volume': df['y_true'],
                            'openinterest' : df['y_true']})
price_df.to_csv('price.csv', index=False)

# generate pos_path data to be feed into backtest bot 
pos_df = pd.DataFrame({'Timeframe': df['timestamp'],
                        'high': df['action']})
pos_df.to_csv('pos.csv', index=False)
