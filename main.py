from keras import Model
from keras.models import Sequential
from keras.models import save_model
from keras.models import load_model
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import GRU
import keras.backend import K
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


import matplotlib.pyplot as plt
import pandas as pd
import pickle
import datetime
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib
from numpy import array
from data_preprocessor import * 
import numpy as np
import sys

# Make our plot a bit formal
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

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

def build_model(type, window_size, columns_num):
    if(type == 'LSTM'):
        nn_model = Sequential()
        nn_model.add(LSTM(50, activation='relu', input_shape = (window_size, columns_num)))
        nn_model.add(Dense(units = 1))
    if(type == 'GRU'):
        nn_model = Sequential()
        nn_model.add(GRU(50, activation='relu', input_shape = (window_size, columns_num)))
        nn_model.add(Dense(units = 1))
    return nn_model


def create_LSTM_with_attention(hidden_units, dense_units, input_shape, activation):
    x=Input(shape=input_shape)
    LSTM_layer = LSTM(50, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(LSTM_layer)
    outputs=Dense(1, trainable=True, activation=activation)(attention_layer)
    model=Model(x,outputs)
    model.compile(loss='mse', optimizer='adam')    
    return model


def my_dump(obj, fname): 
    try:
        fp=open(fname + '.pydata','wb')
        pickle.dump(obj,fp)
        fp.close()
    except Exception as e:
        print(e)
        
def main():
    n_timestamp = 300
    n_epochs = 5
    
    starttime_0 = datetime.datetime.now()
    
    df = load_data('2021-07-31')
    preprocessor = data_preprocessor(df, n_timestamp)
    f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
    #print('############DEBUG############')
    #print(f1.shape)
    X = preprocessor.generate_X(f1, f2, f3, f4, f5)
    y = preprocessor.generate_y(f1)
    #print(y.shape)
    X_train, X_test, y_train, y_test, sc, test_idx = preprocessor.train_test_split(X, y)
    #print(df['timestamp'][test_idx + n_timestamp:].shape)
    #print('############DEBUG############')
    
    model = build_model('LSTM', X_train.shape[1])
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
    starttime_1 = datetime.datetime.now()
    history = model.fit(X_train, y_train.reshape(-1,1), epochs = n_epochs,
    batch_size = 200)
    loss = history.history['loss']
    epochs = range(len(loss))
    endtime_1 = datetime.datetime.now()
    
    
    #
    # Get predicted data
    #
    y_predicted = model.predict(X_test)
    
    #
    # 'De-normalize' the data
    #
    y_predicted_descaled = sc.inverse_transform(y_predicted)
    y_test_descaled = sc.inverse_transform(y_test.reshape(-1,1))
    
    #save_model(model, 'LSTM_1800t.model')
    #joblib.dump(sc, 'scaler.pydata')
    
    endtime_0 = datetime.datetime.now()
    print('The whole running time:')
    print((endtime_0 - starttime_0).seconds)
    
    plt.figure()
    plt.plot(df['timestamp'][test_idx + n_timestamp + 2000 : test_idx + n_timestamp + 2100], y_predicted_descaled.flatten()[2000:2100], color = 'black', linewidth=0.5, label = 'Predicted value')
    plt.plot(df['timestamp'][test_idx + n_timestamp + 2000 : test_idx + n_timestamp + 2100], y_test_descaled.flatten()[2000:2100], color = 'red', linewidth=0.5, label = 'True value')
    plt.ylabel('Mid-Price')
    plt.xlabel('Timestamp')
    plt.legend()
    plt.title('1800-timestamp Prediction (LSTM)')
    plt.savefig('1800_timestamp_LSTM')
    plt.close()
    
    plt.figure()
    plt.plot(epochs, loss, color = 'black')
    plt.ylabel('Mid-Price')
    plt.xlabel('Timestamp')
    plt.title('Traning Curve (1800 timestamps, LSTM)')
    plt.savefig('training_curve_1800_LSTM')
    plt.close()
    
    mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
    r2 = r2_score(y_test_descaled, y_predicted_descaled)
    print("mse=" + str(round(mse,2)))
    print("r2=" + str(round(r2,2)))
    
    # Get time-series data for back testing
    timestamp_arr = np.array(df[['timestamp']][test_idx + n_timestamp:]).flatten()
    y_predicted_timeseries = pd.DataFrame({'timestamp': timestamp_arr, 'y_predicted': y_predicted_descaled.flatten()})
    my_dump(y_predicted_timeseries, 'y_predicted')
    
    with open('log.txt','a') as f:
        f.write('[' + str(datetime.datetime.now().replace(microsecond=0)) + ']')
        f.write('\tGRU model, n_timestamp = ' + str(n_timestamp) + ', n_epochs = ' + str(n_epochs))
        f.write('\tmse = ' + str(round(mse,2)) + ', re_score = ' + str(round(r2,2)))
            
        #temp_1 = np.array([timestamp_arr[i] for i in range(0, len(timestamp_arr), interval)])
        #temp_2 = np.array([y_predicted_descaled[0][i] for i in range(0, len(timestamp_arr), interval)])
        #y_pred_df = pd.DataFrame({'timestamp': temp_1, 'y_predicted': temp_2})

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

def train_with_one_day(date, window_size, n_timestamp):
    n_epochs = 5
    df = load_data(date)
    preprocessor = data_preprocessor(df, n_timestamp)
    f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
    X = preprocessor.generate_X(f1, f2, f3, f4, f5)
    y = preprocessor.generate_y(f1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1,1))[:,0]
    model = build_model('GRU', window_size, X.shape[1])
    print(model.summary())
    '''
    hidden_units = 3
    model = create_LSTM_with_attention(hidden_units=hidden_units, dense_units=1, input_shape=(window_size, X.shape[1]), activation='tanh')
    '''
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # X = np.reshape(X, (X.shape[0],1,X.shape[1]))
    X = X[:-(X.shape[0]%window_size)]
    X = np.reshape(X, (X.shape[0]//window_size, window_size, X.shape[1]))
    y = y[:-(y.shape[0]%window_size)]
    y = np.reshape(y, (y.shape[0]//window_size, window_size))
    #y_input = y.flatten()
    #y_input = np.array([y_input[i] for i in range(n_timestamp, len(y_input), n_timestamp)])
    history = model.fit(X, y, epochs = n_epochs, batch_size = 200)
    loss = history.history['loss']
    epochs = range(len(loss))
    save_model(model, 'GRU_fear_300t.model')
    # joblib.dump(scaler, 'scaler.pydata')
    '''
    plt.figure()
    plt.plot(epochs, loss, color = 'black')
    plt.ylabel('MSE LOSS')
    plt.xlabel('epoch')
    plt.title('Traning Curve (GRU_300timestamps)')
    plt.savefig('training_curve_300_GRU')
    plt.close()
    '''
    
    
    
    
if __name__ == '__main__':
    window_size = 30
    n_timestamp = 300
    if(len(sys.argv)==1):
        main()
    elif(sys.argv[1]=='-t'):
        date = sys.argv[2] # 2021-07-30
        train_with_one_day(date, window_size, n_timestamp)
    elif(sys.argv[1]=='-p'):
        model_path = sys.argv[2]
        date = sys.argv[3]
        model = load_model(model_path)
        df = load_data(date)
        df = df.iloc[:5000]
        preprocessor = data_preprocessor(df, n_timestamp)
        f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
        X = preprocessor.generate_X(f1, f2, f3, f4, f5)
        y_origin = preprocessor.generate_y(f1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(y_origin.reshape(-1,1))
        y_true = y_origin
        if(X.shape[0]%window_size != 0 ):
            X = X[:-(X.shape[0]%window_size)]
        print(X.shape)
        print(X.shape[0]//window_size)
        X = np.reshape(X, (X.shape[0]//window_size, window_size, X.shape[1]))
        y_true = np.array([y_origin[i] for i in range(window_size, len(y_origin), window_size)])
        y_predicted = model.predict(X)
        y_predicted_descaled = scaler.inverse_transform(y_predicted)
        plt.figure()
        y_pred = y_predicted_descaled.flatten()
        # y_interval = np.array([y_pred[i] for i in range(0, len(y_pred), 100)])
        timestamp_interval = np.array([df['timestamp'][i] for i in range(n_timestamp + window_size, len(df['timestamp']), window_size)])
        plt.plot(timestamp_interval, y_pred, color = 'black', linewidth=0.5, label = 'Predicted value')
        plt.plot(df['timestamp'][n_timestamp : ], y_origin, color = 'red', linewidth=0.5, label = 'True value')
        plt.ylabel('Mid-Price')
        plt.xlabel('Timestamp')
        plt.legend()
        plt.title('GRU 300-timestamp Prediction (F - G)')
        plt.savefig('300_GRU_0731')
        plt.close()
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # print("mse=" + str(round(mse,2)))
        print("r2=" + str(round(r2,2)))
        
    elif(sys.argv[1] == '-ps'):
        model_path = sys.argv[2]
        date = sys.argv[3]
        model = load_model(model_path)
        df = load_data(date)
        df = df.iloc[:5000]
        preprocessor = data_preprocessor(df, n_timestamp)
        f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
        X = preprocessor.generate_X(f1, f2, f3, f4, f5)
        y_origin = preprocessor.generate_y(f1)
        if(X.shape[0]%window_size != 0 ):
            X = X[:-(X.shape[0]%window_size)]
            y = y_origin[:-(y_origin.shape[0]%window_size)]
        X_copy = np.array_split(X,X.shape[0]//window_size)
        y_copy = np.array_split(y,X.shape[0]//window_size)
        y_return_list = []
        for i in range(len(X_copy)):
            X = X_copy[i]
            y = y_copy[i]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit_transform(y.reshape(-1,1))
            X = np.reshape(X, (1, window_size, X.shape[1]))
            y_predicted = model.predict(X)
            y_predicted_descaled = scaler.inverse_transform(y_predicted)
            y_return = y_predicted_descaled[len(y_predicted_descaled)-1][0]
            y_return_list.append(y_return)
        
        
        y_coord = np.array(y_return_list).reshape(-1,1)
        x_coord = np.array([df['timestamp'][i] for i in range(n_timestamp + window_size, len(df['timestamp']), window_size)]).reshape(-1,1)
        pred_points = np.concatenate((x_coord, y_coord), axis = 1)
        
        x_coord = np.array([df['timestamp'][i] for i in range(window_size, len(df['timestamp']), window_size)]).reshape(-1,1)
        y_coord = np.array([y_origin[i] for i in range(0, y_origin.shape[0], window_size)]).reshape(-1,1)
        true_points = np.concatenate((x_coord, y_coord), axis = 1)
        true_points = true_points[:-1]
        print(pred_points.shape)
        print(true_points.shape)
            
        plt.figure()
        # y_interval = np.array([y_pred[i] for i in range(0, len(y_pred), 100)])
        timestamp_interval = np.array([df['timestamp'][i] for i in range(n_timestamp + window_size, len(df['timestamp']), window_size)])
        plt.plot(df['timestamp'][n_timestamp : ], y_origin, color = 'red', linewidth=0.5, label = 'True value')
        plt.ylabel('Mid-Price')
        plt.xlabel('Timestamp')
        plt.legend()
        plt.title('300-timestamp Prediction (F-F)')
        plt.savefig('ff_300_GRU')
        plt.close()
    
