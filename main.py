from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt



import matplotlib.pyplot as plt
import pandas as pd
import pickle
import datetime
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array
from data_preprocessor import * 
import numpy as np


from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

import datetime

n_timestamp = 50
n_epochs = 8

# make plot formal
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
'''
def load_data(date):
    df = pd.read_csv('../order_book/binance-futures_book_snapshot_5_' + date + '_BTCUSDT.csv.gz',
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
'''
def load_data(dates):
    all_df = []
    for date in dates:
        chunk = pd.read_csv('../order_book/binance-futures_book_snapshot_5_' + date + '_BTCUSDT.csv.gz',
                        header = 0,
                        names = ['timestamp', 'Pa1', 'Va1', 'Pb1', 'Vb1', 'Pa2', 'Va2', 'Pb2', 'Vb2', 
                                'Pa3', 'Va3', 'Pb3', 'Vb3', 'Pa4', 'Va4', 'Pb4', 'Vb4', 
                                'Pa5', 'Va5', 'Pb5', 'Vb5'],
                        usecols = [2] + list(range(4, 24)),
                        compression = 'gzip',
                        chunksize=1000000
                        )
        df = pd.concat(chunk)
        df['timestamp'] = [str(x)[:-6]+'.'+str(x)[-6:] for x in df['timestamp']]
        df['timestamp'] = [dt.fromtimestamp(float(x)) for x in df['timestamp']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        all_df.append(df)
    all_df_= pd.concat(all_df)
    return all_df_


def build_model(type):
    if(type == 'LSTM'):
        nn_model = Sequential()
        nn_model.add(LSTM(50, activation='relu', input_shape = (1, X_train.shape[1])))
        nn_model.add(Dense(units = 1))
    if(type == 'GRU'):
        nn_model = Sequential()
        nn_model.add(GRU(50, activation='relu', input_shape = (1, X_train.shape[1])))
        nn_model.add(Dense(units = 1))
    return nn_model

def my_dump(obj, fname): 
    try:
        fp=open(fname + '.pydata','wb')
        pickle.dump(obj,fp)
        fp.close()
    except Exception as e:
        print(e)



starttime_0 = datetime.datetime.now()

# data preprocess

start = datetime.datetime.strptime("29-10-2021", "%d-%m-%Y")
end = datetime.datetime.strptime("29-11-2021", "%d-%m-%Y")
date_temp = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_generated = [d.strftime("%Y-%m-%d") for d in date_temp]
df = load_data(date_generated)
print(df)

preprocessor = data_preprocessor(df, n_timestamp)
f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
X = preprocessor.generate_X(f1, f2, f3, f4, f5)
y = preprocessor.generate_y(f1)
X_train, X_test, y_train, y_test, sc = preprocessor.train_test_split(X, y)

# build model
model = build_model('GRU')
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))

# train on model
starttime_1 = datetime.datetime.now()
history = model.fit(X_train, y_train.reshape(-1,1), epochs = n_epochs, batch_size = 200)
loss = history.history['loss']
epochs = range(len(loss))

# record the tarining time used in seconds
endtime_1 = datetime.datetime.now()
with open('time_log.txt','w') as f:
	f.write('The time to train the LSTM model with GPU: '+ str((endtime_1-starttime_1).seconds))


# to predict the data
y_predicted = model.predict(X_test)

# denormalize the predicted_y to the price 
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train.reshape(-1,1))
y_test_descaled = sc.inverse_transform(y_test.reshape(-1,1))
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()

# dump model and history 
my_dump(model, 'model')
my_dump(history, 'history')

# record the whole running time used in seconds
endtime_0 = datetime.datetime.now()
print((endtime_0 - starttime_0).seconds)
print('The whole running time:')



'''
# set input number of timestamps and training days
n_timestamp = 10
train_days = 1500  # number of days to train from
testing_days = 500 # number of days to be predicted
n_epochs = 25
filter_on = 1
'''



mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse,2)))
print("r2=" + str(round(r2,2)))
