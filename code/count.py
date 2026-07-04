import pandas as pd

file_path = '../data/secom_dataset.xlsx'
new_file_path = '../data/secom_deleted_dataset.xlsx'

df = pd.read_excel(file_path,sheet_name='SECOM_Data')

missing_ratio = df.isnull().sum(axis=0)/len(df)

bad_columns = missing_ratio[missing_ratio >= 0.5].index

df_cleaned = df.drop(columns=bad_columns)

df_cleaned.to_excel(new_file_path,sheet_name='SECOM_Data',index=False)

