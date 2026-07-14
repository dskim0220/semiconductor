import pandas as pd
import numpy as np

file_path = '../data/secom_imputed_dataset.xlsx'

def prepare_data(df, target_column):
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    return X, y

def normalize(values):
    min = min(values)
    max = max(values)
    return [(v-min) / (max-min) for v in values]

def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradient_descent(W, b, X_batch, y_batch, rate):
    #X_batch: B * n
    #y_batch: B * 1
    #W: n * 1
    batch_size = len(X_batch)
    z = np.dot(X_batch, W) + b
    y_hat = sigmoid(z)
    error = y_hat - y_batch
    
    dw = (1/batch_size) * np.dot(X_batch.T, error)
    db = np.mean(error)
    
    W -= rate * dw
    b -= rate * db
    
    return W, b
    
    
    
    

    
    
    
    
    
    
    
       

