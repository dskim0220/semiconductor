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

def gradient_descent(W, b, X_batch, y_batch, batch):
    
    

    
    
    
    
    
    
    
       

