import pandas as pd
import numpy as np

file_path = '../data/secom_deleted_dataset.xlsx'
new_path = '../data/secom_imputed_dataset.xlsx'

df_cleaned = pd.read_excel(file_path,sheet_name = 'SECOM_Data')

y = df_cleaned['result']
x = df_cleaned.drop(columns=['result'])

x_filled = x.fillna(x.median())

df = pd.concat([y,x_filled],axis=1)

df.to_excel(new_path,sheet_name='SECOM_Data',index=False)





    