import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pandas_datareader as data
import yfinance as yf
# import keras.models
from tensorflow.keras.models import load_model
import streamlit as st
start='2010-01-01'
end='2024-10-01'
st.title('Stock Predictor')
input=st.text_input('Enter stock ticker','AAPL')
df=yf.download(input,start=start, end=end)
st.subheader('Data from 2010 to 2024')
st.write(df.describe())
st.subheader('Closing Price vs  Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,label='Stock')
st.pyplot(fig)
st.subheader('Opening Price vs  Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Open,label='Stock')
st.pyplot(fig)
st.subheader('Closing Price vs  Time chart MA7')
ma7=df.Close.rolling(7).mean()
fig1=plt.figure(figsize=(12,6))
plt.plot(df.Close,label='Stock')
plt.plot(ma7,label='Moving Average100')
plt.legend()
st.pyplot(fig1)

st.subheader('Closing Price vs  Time chart MA 200 and MA100')
ma21=df.Close.rolling(21).mean()
fig2=plt.figure(figsize=(12,6))
plt.plot(df.Close,label='Stock')
plt.plot(ma7,label='Moving Average100')
plt.plot(ma21,label='Moving Average200')
plt.legend()
st.pyplot(fig2)
#Train and test
train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

#Training transform


model=load_model(r"C:\Users\ravik\PycharmProjects\Stock\stockpred.keras")

# data_training=scaler.fit_transform(train)
# xtrain=[]
# ytrain=[]
# for i in range(100,data_training.shape[0]):
#     xtrain.append(data_training[i-100:i])
#     ytrain.append(data_training[i,0])
# xtrain,ytrain=np.array(xtrain),np.array(ytrain)

#testing transform
past7=train.tail(7)
finaldf = pd.concat([past7, test], ignore_index=True)
inputdata=scaler.fit_transform(finaldf)

xtest=[]
ytest=[]
for i in range(7,inputdata.shape[0]):
    xtest.append(inputdata[i-7:i])
    ytest.append(inputdata[i,0])
xtest,ytest=np.array(xtest),np.array(ytest)

ypred=model.predict(xtest)

scale=scaler.scale_
scalefactor=1/scale[0]
ypred=ypred*scalefactor
ytest=ytest*scalefactor


st.subheader('Prediction vs Original')
fig3=plt.figure(figsize=(12,6))
plt.plot(ytest,'b',label='Original Price')
plt.plot(ypred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)