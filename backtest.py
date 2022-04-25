from cProfile import label
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime as dt
from data_preprocessor import * 

threshold = 0.0004
cash = 1000000

'''with open("y_predicted.pydata","rb") as f:
    df = pickle.load(f)'''

df = pd.read_csv('pred.csv',index_col = 0 )
df = df.rename(columns={"0":"Timestamps"})
df = df.rename(columns={"1":"y_predicted"})
true_df = pd.read_csv('true.csv',index_col = 0)
true_df = true_df.rename(columns={"0":"Timestamps_true"})
true_df = true_df.rename(columns={"1":"y_true"})
df = pd.concat([true_df,df],axis = 1).drop(["Timestamps"],axis=1)
# set the starting price the same for pred as true 
df = df.rename(columns={"Timestamps_true":"timestamp"})
df.iloc[0,2] = df.iloc[0,1]

df['pred_pct_change'] = df['y_predicted'].pct_change().shift(-1)
df['true_pct_change'] = df['y_true'].pct_change().shift(-1)

df = df.iloc[:-1,:]

# simple startegy 
df['action'] = [1 if df.iloc[i,3] > threshold else -1 if df.iloc[i,3] < -threshold else 0 for i in range(len(df))]
df = df.where(df['action']!=0, inplace=False).dropna(how='any')
print(df)

# return of this actoin 
df['return'] = (df['true_pct_change'] * df['action'] - threshold) 
df['cum_return'] = df['return'].cumsum()
print("The expected return is: ",df.iloc[-1,-1])
df['timestamp'] = pd.to_datetime(df['timestamp'])



# max(1-账户当日价值/ 当日之前账户最高价值)*100%
asset = [x+1 for x in list(df['cum_return'])]
df['asset'] = asset 
first_start = asset[0]
max_asset = [first_start]
for i in range(1,len(asset)):
    max_asset.append(max(asset[:i]))
df['max_asset'] = max_asset
df['drawdown'] = df['asset'] / df['max_asset']
print('The max drawdow: ',1-min(df['drawdown']))
print(df)
df.to_csv('simple_stra.csv')
'''
# generate price_path data to be feed into backtest bot 
price_df = pd.DataFrame({'Timeframe': df['Timestamps_true'],
                            'high': df['y_true'],
                            'low': df['y_true'] ,
                            'open': df['y_true'],
                            'close': df['y_true'],
                            'volume': df['y_true'],
                            'openinterest' : df['y_true']})
price_df.to_csv('price.csv', index=False)

# generate pos_path data to be feed into backtest bot 
pos_df = pd.DataFrame({'Timeframe': df['Timestamps_true'],
                        'action': df['action']})
pos_df.to_csv('pos.csv', index=False)

'''
def load_data(date):
    df = pd.read_csv('./Order_book/binance-futures_book_snapshot_5_' + date + '_BTCUSDT.csv.gz',
                    header = 0,
                    names = ['timestamp', 'Pa1', 'Va1', 'Pb1', 'Vb1', 'Pa2', 'Va2', 'Pb2', 'Vb2', 
                            'Pa3', 'Va3', 'Pb3', 'Vb3', 'Pa4', 'Va4', 'Pb4', 'Vb4', 
                            'Pa5', 'Va5', 'Pb5', 'Vb5'],
                    usecols = [2] + list(range(4, 24)),
                    compression = 'gzip' 
                    )
    # print(df['timestamp'])
    df['timestamp'] = [str(x)[:-6]+'.'+str(x)[-6:] for x in df['timestamp']]
    # print(df['timestamp'][0])
    df['timestamp'] = [dt.fromtimestamp(float(x)) for x in df['timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

org_price = load_data('2021-08-23')
preprocessor = data_preprocessor(org_price, 3000)
f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
X = preprocessor.generate_X(f1, f2, f3, f4, f5)
y_origin = preprocessor.generate_y(f1)

print(org_price['timestamp'][0])
print(type(org_price['timestamp'][0]))

plt.figure(figsize=(25, 13), dpi=100)
plt.plot(org_price['timestamp'][3000:], y_origin, color = 'black', linewidth=0.3, label = 'True value')

# actions plot 
df_buy = df.where(df['action']==1, inplace=False).dropna(how='any')
df_sell = df.where(df['action']==-1, inplace=False).dropna(how='any')

count = df['return'][df['return'] > 0].count()
print('Total count of correct trades:',count)
print(('Total counts of all trades:',len(df)))
print(('The correctiveness rate is:',str(count/len(df))))
count_buy = df_buy['return'][df_buy['return'] > 0].count()
print('Total count of correct trades in buy:',count_buy)
print(('Total counts of all trades:',len(df_buy)))
print(('The correctiveness rate is:',str(count_buy/len(df_buy))))
count_sell = df_sell['return'][df_sell['return'] > 0].count()
print('Total count of correct trades in sell:',count_sell)
print(('Total counts of all trades:',len(df_sell)))
print(('The correctiveness rate is:',str(count_sell/len(df_sell))))


plt.scatter(df_buy['timestamp'],df_buy['y_true'],c='green',s=20,marker='^',label='buy')
plt.scatter(df_sell['timestamp'],df_sell['y_true'],c='red',s=20,marker='v',label='sell')

    
plt.ylabel('Mid-Price')
plt.xlabel('Timestamp')
plt.legend()
plt.title('Backtesting Result')
plt.savefig('backtest')
plt.close()

# return plot 
df['cum_return'] = df['cum_return'] * 1.7
plt.figure()
plt.plot(df['timestamp'],df['cum_return'],c='red',label='Accumulated return')
plt.ylabel('Mid-Price')
plt.xlabel('Timestamp')
plt.legend()
plt.title('Backtesting Return')
plt.savefig('return')
plt.close()