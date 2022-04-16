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
from sklearn.externals import joblib
from numpy import array
from data_preprocessor import * 
import numpy as np

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

import datetime

n_timestamp = 1800
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
        chunk = pd.read_csv('./order_book/binance-futures_book_snapshot_5_' + date + '_BTCUSDT.csv.gz',
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

start = datetime.datetime.strptime("30-10-2021", "%d-%m-%Y")
end = datetime.datetime.strptime("31-10-2021", "%d-%m-%Y")
date_temp = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_generated = [d.strftime("%Y-%m-%d") for d in date_temp]
df = load_data(date_generated)
print(df)

preprocessor = data_preprocessor(df, n_timestamp)
f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
X = preprocessor.generate_X(f1, f2, f3, f4, f5)
y = preprocessor.generate_y(f1)
X_train, X_test, y_train, y_test, sc, test_idx = preprocessor.train_test_split(X, y)

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

# record the training time used in seconds
endtime_1 = datetime.datetime.now()

# to predict the data
y_predicted = model.predict(X_test)

# denormalize the predicted_y to the price 
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train.reshape(-1,1))
y_test_descaled = sc.inverse_transform(y_test.reshape(-1,1))
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()

# save model, scaler 
save_model(model, 'GRU_1800t.model')
joblib.dump(sc, 'scaler.pydata')
# my_dump(history.history, 'history')

# record the whole running time used in seconds
endtime_0 = datetime.datetime.now()
print('The whole running time:')
print((endtime_0 - starttime_0).seconds)

plt.figure()
plt.plot(df['timestamp'][test_idx + n_timestamp : test_idx + n_timestamp + 100], y_predicted_descaled.flatten()[:100], color = 'black', linewidth=0.5, label = 'Predicted value')
plt.plot(df['timestamp'][test_idx + n_timestamp : test_idx + n_timestamp + 100], y_test_descaled.flatten()[:100], color = 'red', linewidth=0.5, label = 'True value')
plt.ylabel('Mid-Price')
plt.xlabel('Timestamp')
plt.legend()
plt.title('1800-timestamp Prediction')
plt.savefig('1800_timestamp')
plt.close()

mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse,2)))
print("r2=" + str(round(r2,2)))

# Get time-series data for back testing
timestamp_arr = np.array(df[['timestamp']][test_idx + n_timestamp:]).flatten()
y_predicted_timeseries = pd.DataFrame({'timestamp': timestamp_arr, 'y_predicted': y_predicted_descaled.flatten(), 'y_true':y_test_descaled.flatten()})
my_dump(y_predicted_timeseries, 'y_predicted')

with open('log.txt','a') as f:
    f.write('[' + str(datetime.datetime.now().replace(microsecond=0)) + ']')
    f.write('\tGRU model, n_timestamp = ' + str(n_timestamp) + ', n_epochs = ' + str(n_epochs))
    f.write('\tmse = ' + str(round(mse,2)) + ', re_score = ' + str(round(r2,2)))
    
###
# Process To-Be-Predicted Data 
###
def input_data_predict(csv_file_path, model, scaler):
    # read csv
    df = pd.read_csv(csv_file_path, header = None,
                    names = ['Pa1', 'Va1', 'Pb1', 'Vb1', 'Pa2', 'Va2', 'Pb2', 'Vb2', 
                            'Pa3', 'Va3', 'Pb3', 'Vb3', 'Pa4', 'Va4', 'Pb4', 'Vb4', 
                            'Pa5', 'Va5', 'Pb5', 'Vb5', 'timestamp'],
                    compression = 'gzip' 
                    )
    df['timestamp'] = [str(x)[:-6]+'.'+str(x)[-6:] for x in df['timestamp']]
    df['timestamp'] = [dt.fromtimestamp(float(x)) for x in df['timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    preprocessor = data_preprocessor(df);
    f1, f2, f3, f4, f5 = preprocessor.timepoint_feature()
    X = preprocessor.generate_input_X(f1, f2, f3, f4, f5)
    y_predicted = model.predict(X)
    y_predicted = scaler.inverse_transform(y_predicted).flatten()
    
    return y_predicted[len(y_predicted)-1]
