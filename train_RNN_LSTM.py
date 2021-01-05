import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

"""
#read the dataset provided
read_dataset = pd.read_csv("./data/q2_dataset.csv")
read_dataset['Date'] =pd.to_datetime(read_dataset.Date)
read_dataset=read_dataset.sort_values('Date')

dataset = read_dataset.iloc[:, 2:6].values

data=np.empty((1256,13),dtype=float)    #numpy array of index, followed by 12 features
target=np.empty((1256),dtype=float)     #numpy array of target values as open values for 4th day
#creating index "ind" for data to later sort data by date for plotting
ind=1
#converting the data into a set of data and target arrays
for i in range(0,data.shape[0]):
    row=[ind]
    target[i]=dataset[i+3][1]
    for k in range(3):
        for j in range(4):
            row.append(dataset[k+i][j])
    data[i]=row
    ind+=1

#split data into 70% train and 30% test 
data_train,data_test,target_train ,target_test= train_test_split(data,target,test_size=0.30, shuffle=True)
#print(data_train.shape, data_test.shape, target_train.shape,target_test.shape)

#creating train and test numpy arrays having 14 values in each row: 1 index + 12 features + 1 target value
save_train_data=np.empty((data_train.shape[0],14))
save_test_data=np.empty((data_test.shape[0],14))

for i in range (save_train_data.shape[0]):
    row=data_train[i]
    row=np.append(row,target_train[i])
    save_train_data[i]=row
print("train shape",save_train_data.shape)

for i in range (save_test_data.shape[0]):
    row=data_test[i]
    row=np.append(row,target_test[i])
    save_test_data[i]=row
print("test shape",save_test_data.shape)

#saving the data in .csv files
np.savetxt('/content/drive/My Drive/train_data_RNN.csv', [p for p in (save_train_data)], delimiter=',', fmt='%s')
np.savetxt('/content/drive/My Drive/test_data_RNN.csv', [p for p in (save_test_data)], delimiter=',', fmt='%s')
"""
#read train data 
read_train_dataset = pd.read_csv("./data/train_data_RNN.csv",header=None)
train_dataset=read_train_dataset.iloc[:, 1:14].values

#scaling the data using MinMaxScaler
sc = MinMaxScaler()
train_dataset_scaled= sc.fit_transform(train_dataset)

#splitting train data into x_train and y_train
x=np.empty((879,12),dtype=float)
y_train=np.empty((879),dtype=float)

for i in range(train_dataset_scaled.shape[0]):
    x[i]=train_dataset_scaled[i,0:12]
    y_train[i]=train_dataset_scaled[i,12:]

#reshaping x data into (879,3,4) from (879,12)
x_train=x.reshape(879,3,4)

#defining LSTM Model

model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
print(model.summary())
#compile and fit the LSTM model on train data
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1)
#save the model 
model.save('./models/RNN_model.h5')

#printinh final train accuracy and loss value
print("Final training accuracy:", history.history['accuracy'][-1]*100)
print("Final training loss:", history.history['loss'][-1])
