import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import pickle


class data_preprocessor: 
    def __init__(self, df, n_timestamp):
        self.df = df 
        self.n_timestamp = n_timestamp
        
    def timpoint_feature(self):
        # Feature Set V1
        f1 = self.df[['Pa1', 'Pa2', 'Pa3', 'Pa4', 'Pa5', 
                    'Va1', 'Va2', 'Va3', 'Va4', 'Va5', 
                    'Pb1', 'Pb2', 'Pb3', 'Pb4', 'Pb5', 
                    'Vb1', 'Vb2', 'Vb3', 'Vb4', 'Vb5', ]]
        f1 = np.array(f1)

        # Feature Set V2
        temp1 = f1[:, 0:5] - f1[:, 10:15]
        temp2 = (f1[:, 0:5] + f1[:, 10:15]) * 0.5
        f2 = np.concatenate((temp1, temp2), axis = 1)

        # Feature Set V3
        temp1 = (f1[:, 4] - f1[:, 0]).reshape(-1, 1)
        temp2 = (f1[:, 10] - f1[:, 14]).reshape(-1, 1)
        temp3 = abs(f1[:, 1:5] - f1[:, 0:4])
        temp4 = abs(f1[:, 11:15] - f1[:, 10:14])
        f3 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        # Feature Set V4: mean prices and volumes
        temp1 = np.mean(f1[:, :5], axis = 1).reshape(-1, 1)
        temp2 = np.mean(f1[:, 10:15], axis = 1).reshape(-1, 1)
        temp3 = np.mean(f1[:, 5:10], axis = 1).reshape(-1, 1)
        temp4 = np.mean(f1[:, 15:], axis = 1).reshape(-1, 1)
        f4 = np.concatenate((temp1, temp2, temp3, temp4), axis = 1)

        # Feature Set V5: accumulated differences
        temp1 = np.sum(f2[:, 0:5], axis = 1).reshape(-1, 1)
        temp2 = np.sum(f1[:, 5:10] - f1[:, 15:20], axis = 1).reshape(-1, 1)
        f5 = np.concatenate((temp1, temp2), axis = 1)

        return f1, f2, f3, f4, f5 
    
    def generate_X(self, f1, f2, f3, f4, f5): 
        def normalize(X):
            m,n = X.shape
            for j in range(n):
                features = X[:, j]
                mean = features.mean(0)
                std = features.std(0)
                if(std != 0):
                    X[:, j] = (features - mean) / std
                else:
                    X[:, j] = 0
            return X
        # Concatenate all features and normalize
        X = np.concatenate((f1, f2, f3, f4, f5), axis = 1)
        X = X[:-(self.n_timestamp)]
        X = normalize(X)
        return X

    def generate_y(self, f1): 
        y = (f1[:, 0] + f1[:, 10]) * 0.5
        return y[self.n_timestamp:]
    
    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_train = scaler.fit_transform(y_train.reshape(-1,1))[:,0]
        y_test = scaler.transform(y_test.reshape(-1,1))[:,0]
        test_idx = X_train.shape[0]
        return X_train, X_test, y_train, y_test, scaler, test_idx