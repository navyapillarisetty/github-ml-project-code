#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data= pd.read_csv(r"C:\Users\Navya Pillarisetty\Energy data 1990 - 2020.csv")
#print(data)
print(data.head())


# In[2]:


print(data["Region"].unique())
print(len(data["Region"].unique()))


# In[3]:


print(data["country"].unique())
print(len(data["country"].unique()))


# In[4]:


print(data.describe())


# In[5]:


print(data.info())


# In[ ]:





# In[6]:


data= data.drop("Region", axis= 1)
correlation= data.corr()
print(correlation)


# In[7]:


countries= data["country"].unique()
years= data["Year"].unique()
print(countries)
print(years)


# In[8]:


groups= data.groupby("country")
#print(len(groups))
country= data["country"].unique()
#print(country)
group=[]
for i in range (len(groups)):
    country[i]= pd.DataFrame(groups.get_group(country[i]))
    group.append(country[i])
#print(group[0])
new=[]
for i in group:
    i= i.drop("country", axis=1)
    i["Natural gas production (bcm)"]= i["Natural gas production (bcm)"].astype(str)
    #i["Natural gas production (bcm)"] = i["Natural gas production (bcm)"].replace ("n.a.", float("nan"))
    i["Natural gas production (bcm)"] = i["Natural gas production (bcm)"].replace ("n.a.", float(0.0))
    i["Natural gas production (bcm)"]= i["Natural gas production (bcm)"].astype(float)
    i["Coal and lignite domestic consumption (Mt)"]= i["Coal and lignite domestic consumption (Mt)"].astype(str)
    #i["Coal and lignite domestic consumption (Mt)"] = i["Coal and lignite domestic consumption (Mt)"].replace ("n.a.", float("nan"))
    i["Coal and lignite domestic consumption (Mt)"] = i["Coal and lignite domestic consumption (Mt)"].replace ("n.a.", float(0.0))
    i["Coal and lignite domestic consumption (Mt)"]= i["Coal and lignite domestic consumption (Mt)"].astype(float)
    i["Share of wind and solar in electricity production (%)"]= i["Share of wind and solar in electricity production (%)"].astype(str)
    #i["Share of wind and solar in electricity production (%)"] = i["Share of wind and solar in electricity production (%)"].replace ("n.a.", float("nan"))
    i["Share of wind and solar in electricity production (%)"] = i["Share of wind and solar in electricity production (%)"].replace ("n.a.", float(0.0))
    i["Share of wind and solar in electricity production (%)"]= i["Share of wind and solar in electricity production (%)"].astype(float)
    i["Coal and lignite production (Mt)"]= i["Coal and lignite production (Mt)"].astype(str)
    #i["Coal and lignite production (Mt)"] = i["Coal and lignite production (Mt)"].replace ("n.a.", float("nan"))
    i["Coal and lignite production (Mt)"] = i["Coal and lignite production (Mt)"].replace ("n.a.", float(0.0))
    i["Coal and lignite production (Mt)"]= i["Coal and lignite production (Mt)"].astype(float)
    new.append(i)
print(new[0])


# In[9]:


# descriptive analysis of CO2 emissions from fuel combustion
import matplotlib.pyplot as plt
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["CO2 emissions from fuel combustion (MtCO2)"], color= "violet",label="CO2_emission", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()
 


# In[10]:


# descriptive analysis of Average CO2 emission factor (tCO2/toe)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Average CO2 emission factor (tCO2/toe)"], color= "indigo",label="Average CO2 emission factor", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[11]:


# descriptive analysis of CO2 intensity at constant purchasing power parities (kCO2/$15p)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["CO2 intensity at constant purchasing power parities (kCO2/$15p)"], color= "blue",label="CO2 intensity at constant purchasing power parities", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[12]:


# descriptive analysis of Total energy production (Mtoe)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Total energy production (Mtoe)"], color= "green",label="Total energy production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[13]:


# descriptive analysis of Total energy consumption (Mtoe)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Total energy consumption (Mtoe)"], color= "orange",label="Total energy consumption", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[14]:


# descriptive analysis of Share of renewables in electricity production (%)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Share of renewables in electricity production (%)"], color= "red",label="Share of renewables in electricity production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[15]:


# descriptive analysis of Share of electricity in total final energy consumption (%)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Share of electricity in total final energy consumption (%)"], color= "pink",label="Share of electricity in total final energy consumption", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[16]:


# descriptive analysis of Oil products domestic consumption (Mt)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Oil products domestic consumption (Mt)"], color= "purple",label="Oil products domestic consumption", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[17]:


# descriptive analysis of Refined oil products production (Mt)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Refined oil products production (Mt)"], color= "magenta",label="Refined oil products production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[18]:


# descriptive analysis of Natural gas production (bcm)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Natural gas production (bcm)"], color= "black",label="Natural gas production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[19]:


# descriptive analysis of Natural gas domestic consumption (bcm)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Natural gas domestic consumption (bcm)"], color= "brown",label="Natural gas domestic consumption", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[20]:


# descriptive analysis of Energy intensity of GDP at constant purchasing power parities (koe/$15p)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Energy intensity of GDP at constant purchasing power parities (koe/$15p)"], color= "green",label="Energy intensity of GDP at constant purchasing power parities", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[21]:


# descriptive analysis of Electricity production (TWh)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Electricity production (TWh)"], color= "red",label="Electricity production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[22]:


# descriptive analysis of Electricity domestic consumption (TWh)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Electricity domestic consumption (TWh)"], color= "blue",label="Electricity domestic consumption", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[23]:


# descriptive analysis of Coal and lignite domestic consumption (Mt)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Coal and lignite domestic consumption (Mt)"], color= "pink",label="Coal and lignite domestic consumption", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[24]:


# descriptive analysis of Share of wind and solar in electricity production (%)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Share of wind and solar in electricity production (%)"], color= "grey",label="Share of wind and solar in electricity production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[25]:


# descriptive analysis of Crude oil production (Mt)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Crude oil production (Mt)"], color= "orange",label="Crude oil production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[26]:


# descriptive analysis of Coal and lignite production (Mt)
for k in range(len(new)):
    plt.figure(figsize=(18,5))
    x_ticks= range(len(new[k]["Year"]))
    plt.plot(x_ticks,new[k]["Coal and lignite production (Mt)"], color= "red",label="Coal and lignite production", marker="o")
    plt.title(countries[k])
    plt.xticks(x_ticks,new[k]["Year"])
    plt.legend()
    plt.show()


# In[27]:


print(new[0].info())


# In[28]:


print(new[0])


# In[29]:


for index,m in enumerate(new):
    if m.isnull().values.any()==False:
        print(index)
#print(new[19])


# In[30]:


new_data=[]
for j in new:
    j= j.drop("Year", axis=1)
    j["Surplus energy production"]= j["Total energy production (Mtoe)"]- j["Total energy consumption (Mtoe)"]
    j= j.drop(columns=["Total energy production (Mtoe)","Total energy consumption (Mtoe)"])
    j["Surplus oil production"]= j["Refined oil products production (Mt)"]- j["Oil products domestic consumption (Mt)"]
    j= j.drop(columns=["Refined oil products production (Mt)","Oil products domestic consumption (Mt)"])
    j["Surplus natural gas production"]= j["Natural gas production (bcm)"]-j["Natural gas domestic consumption (bcm)"]
    j= j.drop(columns=["Natural gas production (bcm)","Natural gas domestic consumption (bcm)"])
    j["Surplus electricity production"]= j["Electricity production (TWh)"]- j["Electricity domestic consumption (TWh)"]
    j= j.drop(columns=["Electricity production (TWh)","Electricity domestic consumption (TWh)"])
    j["Surplus coal and lignite production"]= j["Coal and lignite production (Mt)"]-j["Coal and lignite domestic consumption (Mt)"]
    j= j.drop(columns=["Coal and lignite production (Mt)","Coal and lignite domestic consumption (Mt)"])
    new_data.append(j)
print(new_data[0].info())


# In[31]:


print(new_data[0].corr())


# In[32]:


from sklearn.model_selection import train_test_split

#train_data=[]
#test_data=[]
#for u in new_data:
#    train, test= train_test_split(u, test_size=0.2, random_state= 32)
#    train_data.append(train)
#    test_data.append(test)
#print(len(train_data[0]))
#print(len(test_data[0]))
#print(len(train_data))
#print(len(test_data))


# In[33]:


import seaborn as sns
for o in range(len(new_data)):
    sns.boxplot(new_data[o]["CO2 emissions from fuel combustion (MtCO2)"])
    plt.title(countries[o])
    plt.show()


# In[34]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Average CO2 emission factor (tCO2/toe)"])
    plt.title(countries[o])
    plt.show()


# In[35]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["CO2 intensity at constant purchasing power parities (kCO2/$15p)"])
    plt.title(countries[o])
    plt.show()


# In[36]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Share of renewables in electricity production (%)"])
    plt.title(countries[o])
    plt.show()


# In[37]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Share of electricity in total final energy consumption (%)"])
    plt.title(countries[o])
    plt.show()


# In[38]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Energy intensity of GDP at constant purchasing power parities (koe/$15p)"])
    plt.title(countries[o])
    plt.show()


# In[39]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Share of wind and solar in electricity production (%)"])
    plt.title(countries[o])
    plt.show()


# In[40]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Crude oil production (Mt)"])
    plt.title(countries[o])
    plt.show()


# In[41]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Surplus energy production"])
    plt.title(countries[o])
    plt.show()


# In[42]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Surplus oil production"])
    plt.title(countries[o])
    plt.show()


# In[43]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Surplus natural gas production"])
    plt.title(countries[o])
    plt.show()


# In[44]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Surplus electricity production"])
    plt.title(countries[o])
    plt.show()


# In[45]:


for o in range(len(new_data)):
    sns.boxplot(new_data[o]["Surplus coal and lignite production"])
    plt.title(countries[o])
    plt.show()


# In[46]:


from sklearn.impute import SimpleImputer
median= SimpleImputer(strategy= "median")
columns= new_data[0].columns
#print(len(columns))
count=0
for index, m in enumerate(new_data):
    if m.isnull().values.any()==True:
        print(index)
new=[]
median_list=[]
for n in range(len(new_data)):
    x= median.fit_transform(new_data[n])
    median_list.append(median.statistics_)
    x_= pd.DataFrame(x, columns= columns)
    new.append(x_)
print(new[0])


# In[47]:


from sklearn.preprocessing import MinMaxScaler
minmax= MinMaxScaler()
new_set=[]
for u in range(len(new)):
    scale= minmax.fit_transform(new[u])
    scale_= pd.DataFrame(scale, columns=columns)
    new_set.append(scale_)
print(len(new_set))
print(new_set[0])


# In[48]:


print(new_set[0].info())


# In[49]:


from sklearn.model_selection import train_test_split
train_set=[]
test_set=[]
for u in range(len(new_set)):
    train, test= train_test_split(new_set[u], test_size=0.2, random_state= 32)
    train_set.append(train)
    test_set.append(test)
print(len(train_set[0]))
print(len(test_set[0]))
print(len(train_set))
print(len(test_set))


# In[50]:


import numpy as np
train_set_X= []
train_set_Y=[]
for f in range(len(train_set)):
    train_X= train_set[f].drop("Surplus energy production", axis=1)
    train_Y= train_set[f]["Surplus energy production"]
    train_set_X.append(train_X)
    train_set_Y.append(train_Y)
test_set_X= []
test_set_Y=[]
for g in range(len(test_set)):
    test_X= test_set[g].drop("Surplus energy production", axis=1)
    test_Y= test_set[g]["Surplus energy production"]
    test_set_X.append(test_X)
    test_set_Y.append(test_Y)
print(len(train_set_X))
print(len(train_set_Y))
print(len(test_set_X))
print(len(test_set_Y))
seq_length=1
print(train_set_X[0].shape)
trainX=[]
for i in range(len(train_set_X)):
    arr= train_set_X[i].values
    reshape_arr= arr.reshape(len(train_set_X[i]),seq_length, train_set_X[i].shape[1])
    trainX.append(reshape_arr)
print(trainX[0].shape)
trainY=[]
for j in range(len(train_set_Y)):
    arr= train_set_Y[j].values
    reshape_arr= arr.reshape(len(train_set_Y[j]),seq_length, 1)
    trainY.append(reshape_arr)
print(trainY[0].shape)
testX=[]
for k in range(len(test_set_X)):
    arr= test_set_X[k].values
    reshape_arr= arr.reshape(len(test_set_X[k]),seq_length, test_set_X[k].shape[1])
    testX.append(reshape_arr)
print(testX[0].shape)
testY=[]
for l in range(len(test_set_Y)):
    arr= test_set_Y[l].values
    reshape_arr= arr.reshape(len(test_set_Y[l]),seq_length, 1)
    testY.append(reshape_arr)
print(testY[0].shape)


# In[51]:


# rnn_gru
import tensorflow as tf
from tensorflow import keras
import numpy as np
seq_length=1
pred_values_rnn_gru= []
def rnn_network_gru(data1, data2, data3, data4):
    model= tf.keras.models.Sequential([
        tf.keras.layers.GRU(12,return_sequences= True,unroll= True, input_shape=(1,12)),
        tf.keras.layers.GRU(128, return_sequences= True,unroll=True, dropout= 0.4),
        tf.keras.layers.GRU(64, return_sequences= True, unroll= True),
        tf.keras.layers.GRU(32, return_sequences= True,unroll= True, dropout=0.4),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer= tf.keras.optimizers.Adam())
    history= model.fit(data1, data2, epochs= 15)
    model.evaluate(data3, data4)
    y_new= model.predict(data3)
    pred_values_rnn_gru.append(y_new)
    return y_new
for s in range(len(train_set_X)):
    rnn= rnn_network_gru(trainX[s], trainY[s], testX[s], testY[s])
print(len(pred_values_rnn_gru))


# In[52]:


print(len(pred_values_rnn_gru))
pred_gru=[]
for u in range(len(pred_values_rnn_gru)):
    arr= pred_values_rnn_gru[u].reshape(len(pred_values_rnn_gru[u]))
    pred_gru.append(arr)
print(pred_gru[0].shape)
testY_reshape=[]
for y in range(len(testY)):
    arr= testY[y].reshape(len(testY[0]))
    testY_reshape.append(arr)
from sklearn.metrics import mean_absolute_error
gru_mae=[]
for r in range(len(pred_gru)):
    mae= mean_absolute_error(testY_reshape[r],pred_gru[r])
    gru_mae.append(mae)
print(len(gru_mae))
print(gru_mae)


# In[53]:


#rnn_lstm
seq_length=1
pred_values_lstm= []
def rnn_network_lstm(data1, data2, data3, data4):
    model= tf.keras.models.Sequential([
        tf.keras.layers.LSTM(12,return_sequences= True, unroll= True, input_shape=(1,12)),
        tf.keras.layers.LSTM(128, return_sequences= True, unroll=True, dropout= 0.4),
        tf.keras.layers.LSTM(64, return_sequences= True, unroll= True),
        tf.keras.layers.LSTM(32, return_sequences= True, unroll= True, dropout=0.4),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer= tf.keras.optimizers.Adam())
    history= model.fit(data1, data2, epochs= 15)
    model.evaluate(data3, data4)
    y_new= model.predict(data3)
    pred_values_lstm.append(y_new)
    return y_new
for s in range(len(train_set_X)):
    lstm= rnn_network_lstm(trainX[s], trainY[s], testX[s], testY[s])


# In[54]:


print(len(pred_values_lstm))
pred_lstm=[]
for u in range(len(pred_values_lstm)):
    arr= pred_values_lstm[u].reshape(len(pred_values_lstm[u]))
    pred_lstm.append(arr)
print(pred_lstm[0].shape)
from sklearn.metrics import mean_absolute_error
lstm_mae=[]
for r in range(len(pred_lstm)):
    mae= mean_absolute_error(testY_reshape[r],pred_lstm[r])
    lstm_mae.append(mae)
print(len(lstm_mae))
print(lstm_mae)


# In[55]:


# simple rnn
seq_length=1
pred_values_simple_rnn= []
def rnn_network_simple_rnn(data1, data2, data3, data4):
    model= tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(12,return_sequences= True, unroll= True, input_shape=(1,12)),
        tf.keras.layers.SimpleRNN(128, return_sequences= True, unroll=True, dropout= 0.4),
        tf.keras.layers.SimpleRNN(64, return_sequences= True, unroll= True),
        tf.keras.layers.SimpleRNN(32, return_sequences= True, unroll= True, dropout=0.4),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer= tf.keras.optimizers.Adam())
    history= model.fit(data1, data2, epochs= 15)
    model.evaluate(data3, data4)
    y_new= model.predict(data3)
    pred_values_simple_rnn.append(y_new)
    return y_new
for s in range(len(train_set_X)):
    simple_rnn= rnn_network_simple_rnn(trainX[s], trainY[s], testX[s], testY[s])


# In[56]:


print(len(pred_values_simple_rnn))
pred_simple_rnn=[]
for u in range(len(pred_values_simple_rnn)):
    arr= pred_values_simple_rnn[u].reshape(len(pred_values_simple_rnn[u]))
    pred_simple_rnn.append(arr)
print(pred_simple_rnn[0].shape)
from sklearn.metrics import mean_absolute_error
simple_rnn_mae=[]
for r in range(len(pred_simple_rnn)):
    mae= mean_absolute_error(testY_reshape[r],pred_simple_rnn[r])
    simple_rnn_mae.append(mae)
print(len(simple_rnn_mae))
print(simple_rnn_mae)


# In[57]:


# convolutional network
pred_values_cnn=[]
def cnn_network(data1, data2, data3, data4):
    cnn_model= tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(12, kernel_size= 1, strides=1, dilation_rate= 10, padding= "same", input_shape= (None,12)),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1)
    ])
    cnn_model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer= tf.keras.optimizers.Adam())
    history= cnn_model.fit(data1, data2, epochs= 15)
    cnn_model.evaluate(data3, data4)
    y_new= cnn_model.predict(data3)
    pred_values_cnn.append(y_new)
    return y_new
for s in range(len(train_set_X)):
    cnn= cnn_network(trainX[s], trainY[s], testX[s], testY[s])


# In[58]:


print(len(pred_values_cnn))
pred_cnn=[]
for u in range(len(pred_values_cnn)):
    arr= pred_values_cnn[u].reshape(len(pred_values_cnn[u]))
    pred_cnn.append(arr)
print(pred_cnn[0].shape)
from sklearn.metrics import mean_absolute_error
cnn_mae=[]
for r in range(len(pred_cnn)):
    mae= mean_absolute_error(testY_reshape[r],pred_cnn[r])
    cnn_mae.append(mae)
print(len(cnn_mae))
print(cnn_mae)


# In[59]:


# multilayer perceptron model
pred_values_mlp=[]
def mlp_network(data1, data2, data3, data4):
    mlp_model= tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, input_shape=(None,12)),
        tf.keras.layers.Dense(172),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1)
    ])
    mlp_model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer= tf.keras.optimizers.Adam())
    history= mlp_model.fit(data1, data2, epochs= 15)
    mlp_model.evaluate(data3, data4)
    y_new= mlp_model.predict(data3)
    pred_values_mlp.append(y_new)
    return y_new
for s in range(len(train_set_X)):
    mlp= mlp_network(trainX[s], trainY[s], testX[s], testY[s])


# In[60]:


print(len(pred_values_mlp))
pred_mlp=[]
for u in range(len(pred_values_mlp)):
    arr= pred_values_mlp[u].reshape(len(pred_values_mlp[u]))
    pred_mlp.append(arr)
print(pred_mlp[0].shape)
from sklearn.metrics import mean_absolute_error
mlp_mae=[]
for r in range(len(pred_mlp)):
    mae= mean_absolute_error(testY_reshape[r],pred_mlp[r])
    mlp_mae.append(mae)
print(len(mlp_mae))
print(mlp_mae)


# In[61]:


# xgboost regression
import xgboost
from xgboost import XGBRegressor
pred_xgboost=[]
def xgboost(data1, data2, data3):
    xgboost_model= XGBRegressor(n_estimators= 1000, max_depth= 10, eta= 0.56, subsample=0.8, colsample_bytree=0.9)
    xgboost_model.fit(data1, data2)
    y_pred_xgboost= xgboost_model.predict(data3)
    pred_xgboost.append(y_pred_xgboost)
    return y_pred_xgboost
trainX1=[]
for i in range(len(train_set_X)):
    arr= train_set_X[i].values
    reshape_arr= arr.reshape(len(train_set_X[i]),12)
    trainX1.append(reshape_arr)
print(trainX1[0].shape)
trainY1=[]
for j in range(len(train_set_Y)):
    arr= train_set_Y[j].values
    reshape_arr= arr.reshape(len(train_set_Y[j]), 1)
    trainY1.append(reshape_arr)
print(trainY1[0].shape)
testX1=[]
for k in range(len(test_set_X)):
    arr= test_set_X[k].values
    reshape_arr= arr.reshape(len(test_set_X[k]),12)
    testX1.append(reshape_arr)
print(testX1[0].shape)
for s in range(len(train_set_X)):
    boost= xgboost(trainX1[s],trainY1[s],testX1[s])
#rint(pred_xgboost)


# In[62]:


from sklearn.metrics import mean_absolute_error
xgboost_mae=[]
for r in range(len(pred_xgboost)):
    mae3= mean_absolute_error(testY_reshape[r], pred_xgboost[r])
    xgboost_mae.append(mae3)
print(xgboost_mae)


# In[65]:


# linear regression
from sklearn.linear_model import LinearRegression
pred_linear_reg=[]
def linear_reg(data1,data2,data3):
    linear_model= LinearRegression()
    linear_model.fit(data1,data2)
    linear_pred= linear_model.predict(data3)
    pred_linear_reg.append(linear_pred)
    return linear_pred
for s in range(len(train_set_X)):
    lin= linear_reg(trainX1[s],trainY1[s],testX1[s])


# In[66]:


from sklearn.metrics import mean_absolute_error
linear_reg_mae=[]
for r in range(len(pred_linear_reg)):
    mae3= mean_absolute_error(testY_reshape[r], pred_linear_reg[r])
    linear_reg_mae.append(mae3)
print(linear_reg_mae)


# In[69]:


# ARIMA
import statsmodels.api as sm
pred_arima=[]
def arima(data1):
    arima_model= sm.tsa.arima.ARIMA(data1, order=(0,0,1))
    arima_model = arima_model.fit()
    arima_pred= arima_model.predict()
    pred_arima.append(arima_pred)
    return arima_pred
for s in range(len(train_set_X)):
    lin= arima(testY_reshape[s])


# In[70]:


from sklearn.metrics import mean_absolute_error
arima_mae=[]
for r in range(len(pred_arima)):
    mae3= mean_absolute_error(testY_reshape[r], pred_arima[r])
    arima_mae.append(mae3)
print(arima_mae)


# In[77]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(gru_mae, color="violet", label="gru mae")
plt.plot(lstm_mae, color= "indigo", label="lstm mae")
plt.plot(simple_rnn_mae, color="blue", label= "simple recurrent nn mae")
plt.plot(cnn_mae, color="green", label="convolution nn mae")
plt.plot(mlp_mae, color="black", label="multi layer perceptron mae")
plt.plot(xgboost_mae, color="orange", label="XGBoost regression mae")
plt.plot(linear_reg_mae, color="red", label="linear regression mae")
plt.plot(arima_mae, color="grey", label="ARIMA mae")
plt.legend()
plt.show()


# In[ ]:




