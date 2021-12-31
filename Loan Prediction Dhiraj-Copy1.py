#!/usr/bin/env python
# coding: utf-8

# # Infopillar Solution Pvt Ltd      
# # Task 1 : Loan Prediction Using Machine Learning
# # Author : Dhiraj Mahajan
# # Dataset: http://lib.stat.cmu.edu/datasets/boston

# In[128]:


# Importing Libraries
http://lib.stat.cmu.edu/datasets/boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[129]:


# Loading dataset

boston = load_boston()
print(boston.data)


# In[130]:


boston.data.shape


# In[131]:


# Converting it into Dataframe

dataset=pd.DataFrame(boston.data)


# In[132]:


dataset.columns=boston.feature_names


# In[133]:


dataset['PRICE']=boston.target
print(dataset.head())


# In[134]:


print(boston.target.shape)


# In[135]:


dataset.describe()


# In[136]:


dataset.info()


# In[137]:


dataset.columns


# In[138]:


sns.displot(dataset['PRICE'], color = 'r')


# In[139]:


dataset.isnull().sum()


# In[140]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.hist(dataset['PRICE'],color="green",bins=30)
plt.xlabel("House price in $1000")
plt.show()


# In[141]:


bos_1=pd.DataFrame(boston.data, columns=boston.feature_names)

correlation_matrix=bos_1.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, xticklabels = True, center = None)


# In[142]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[144]:


lm = LinearRegression()
lm.fit(X_train,y_train)

print('Coefficients: \n',lm.coef_)


# In[145]:


lm.fit(X_train,y_train)


# In[146]:


predictions = lm.predict(X_test)
predictions


# In[147]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[148]:


from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
model = lin_model
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
rsquared_score = r2_score(y_test, y_pred)
print('RMSE score:', rmse_score)
print('R2 score:', rsquared_score)
print('\nScatter plot of y_test against y_pred:')
sns.regplot(y_test, y_pred);


# In[149]:


sns.distplot((y_test-predictions),bins=50, color = 'g')
plt.title(' Output')


# In[ ]:





# In[ ]:




