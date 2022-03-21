#
# Core Keras libraries
#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
 
#
# For data conditioning
#
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

#
# Make results reproducible
#
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(1)


# 
# Other essential libraries
#
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import datetime
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array
from data_preprocessor import *

# Make our plot a bit formal
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

def load_data(date):
    df = pd.read_csv('../train_models/Data/Order_book/binance-futures_book_snapshot_5_' + date + '_BTCUSDT.csv.gz',
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

n_timestamp = 10
n_epochs = 25

starttime_0 = datetime.datetime.now()

df = load_data('2021-07-31')
preprocessor = data_preprocessor(df, n_timestamp)
f1, f2, f3, f4, f5 = preprocessor.timpoint_feature()
X = preprocessor.generate_X(f1, f2, f3, f4, f5)
y = preprocessor.generate_y(f1)
X_train, X_test, y_train, y_test, sc = preprocessor.train_test_split(X, y)
model = build_model('GRU')
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)
loss = history.history['loss']
epochs = range(len(loss))

#
# Get predicted data
#
y_predicted = model.predict(X_test)

#
# 'De-normalize' the data
#
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()

my_dump(model, 'model')
my_dump(history, 'history')

endtime_0 = datetime.datetime.now()
print((endtime_0 - starttime_0).seconds)
print('The whole running time:')



'''
#
# Set input number of timestamps and training days
#
n_timestamp = 10
train_days = 1500  # number of days to train from
testing_days = 500 # number of days to be predicted
n_epochs = 25
filter_on = 1




#
# Show results
#
plt.figure(figsize=(8,7))

plt.subplot(3, 1, 1)
plt.plot(dataset['Temperature'], color = 'black', linewidth=1, label = 'True value')
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("All data")


plt.subplot(3, 2, 3)
plt.plot(y_test_descaled, color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled, color = 'red',  linewidth=1, label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (n days)")

plt.subplot(3, 2, 4)
plt.plot(y_test_descaled[0:75], color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled[0:75], color = 'red', label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (first 75 days)")

plt.subplot(3, 3, 7)
plt.plot(epochs, loss, color='black')
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.title("Training curve")

plt.subplot(3, 3, 8)
plt.plot(y_test_descaled-y_predicted_descaled, color='black')
plt.ylabel("Residual")
plt.xlabel("Day")
plt.title("Residual plot")

plt.subplot(3, 3, 9)
plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
plt.ylabel("Y true")
plt.xlabel("Y predicted")
plt.title("Scatter plot")

plt.subplots_adjust(hspace = 0.5, wspace=0.3)
plt.show()



mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse,2)))
print("r2=" + str(round(r2,2)))
'''