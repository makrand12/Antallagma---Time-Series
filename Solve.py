
# coding: utf-8

# In[ ]:

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # We need to predict the data based on Date hence make the Datetime as the index column

# In[ ]:

train_data=pd.read_csv('train.csv',index_col='Datetime',parse_dates=True)
test_data=pd.read_csv('test.csv', index_col='Datetime',parse_dates=True)


# # Dividing the Training Data
# As we can see that we need to predict the Price and the Number of Sales, to get to the result we can see that the Number of sales is directly dependent on the Price of the Item
# Hence Finding the Price of each Item on daily basis can make the prediction of Number of Sales easy 
# 
# Dividing the Training data into Time Series of Price data and the remaining part with Transaction_ID as the index

# In[ ]:

ts=train_data[['Item_ID','Price']]
ts_test=test_data[['Item_ID']]


# In[ ]:

ts_train=ts[['Item_ID']]
ts_target=ts['Price']


# In[ ]:

gbr=GradientBoostingRegressor(learning_rate=0.02,max_depth=50,max_features=1)


# In[ ]:

gbr.fit(ts_train,ts_target)
prediction=gbr.predict(ts_test)


# # Predicted Value is the Price of the Items in next 6 months
# appending the Predicted price to the Test_Data and chaning its Index to Transaction_ID and also droping the Item_ID from it.

# In[ ]:

test_data.set_index('ID',inplace=True)
test_data.drop('Item_ID',1, inplace=True)
test_data['Price']=prediction


# In[ ]:

train_data.set_index('ID',inplace=True)
target=train_data['Number_Of_Sales']
train_data.drop('Item_ID',1,inplace=True)


# In[ ]:

train_data=train_data.fillna(axis=1,method='backfill')


# In[ ]:

gbr=GradientBoostingRegressor(learning_rate=0.02,max_features=4,max_depth=100)
gbr.fit(train_data,target)
sales_pred=gbr.predict(test_data)


# In[ ]:

test_data['Number_Of_Sales']=sales_pred
test_data['Number_Of_Sales']=test_data['Number_Of_Sales'].apply(round).apply(int)


# # Saving the Result
# Saving the Result Dataframe to a csv file for future references

# In[ ]:

result=test_data[['Price','Number_Of_Sales']]
test_data.to_csv("result.csv",index=True)

