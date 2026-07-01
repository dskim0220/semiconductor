import pandas as pd

data_path = '../data/secom.data'
labels_path = '../data/secom_labels.data'

data = pd.read_csv(data_path, sep=" ", header=None)

labels = pd.read_csv(labels_path,sep=" ",header=None,usecols=[0],names=['result'])
labels['result']=labels['result'].map({-1:'Pass',1:'Fail'})
df = pd.concat([labels, data],axis=1)


if 'result' in df.columns:
    col_names = ['result']+[f"Sensor{i}" for i in range(data.shape[1])]
else:
    col_names = [f"Sensor{i}" for i in range(data.shape[1])]
    
df.columns = col_names

output_path = '../data/secom_dataset.xlsx'

df.to_excel(output_path,sheet_name='SECOM_Data',index=False)



