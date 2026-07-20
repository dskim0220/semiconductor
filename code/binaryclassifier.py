import pandas as pd
import numpy as np

file_path = '../data/secom_imputed_dataset.xlsx'

def prepare_data(df, target_column):
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()
    y = y.reshape(-1,1)
    return X, y

def normalize(X):
    min_vals = np.min(X,axis=0)
    max_vals = np.max(X,axis=0)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)

def sigmoid(z):
    z = np.clip(z,-250,250)
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

def model_test(df_test, W, b, train_min, train_max):
    X, y = prepare_data(df_test,'result')
    X_normalized = (X - train_min) / (train_max - train_min + 1e-8)
    
    z = np.dot(X_normalized,W) + b
    y_hat = sigmoid(z)
    
    determined = (y_hat >= 0.5).astype(int)
    accuracy = np.mean(determined == y)
    
    return accuracy
    
    
if __name__== "__main__":
    df = pd.read_excel(file_path,sheet_name='SECOM_Data')
    df['result'] = df['result'].map({'Pass':1, 'Fail':0}) 
    ########
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)
    
    df_train = df[0:1472]
    df_test = df[1472:]
    ########
    X, y = prepare_data(df_train,'result')
    
    train_min = np.min(X,axis=0)
    train_max = np.max(X,axis=0)
    
    X_normalized = normalize(X)
    
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    batch_size = 64
    epochs = 100
    learning_rate = 0.01
    
    W = np.zeros((num_features,1))
    b = 0.0
    
    for epoch in range(epochs):
        
        for i in range(0,num_samples,batch_size):
            
            X_batch = X_normalized[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            W,b = gradient_descent(W,b,X_batch,y_batch,learning_rate)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} 완료")
    
    print("모델 학습이 성공적으로 완료되었습니다!")
    
    accuracy = model_test(df_test,W,b,train_min,train_max)
    print(f"정확도: {accuracy * 100:.2f}%")
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
       

