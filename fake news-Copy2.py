#!/usr/bin/env python
# coding: utf-8

# # Infopillar Solution Pvt Ltd
# # Task 1 : Fake News Detection Project       
# # Author : Dhiraj Mahajan
# # Dataset:  https://bit.ly/3FxCSC4

# In[37]:


# Importing Libraries

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[38]:


import nltk
nltk.download('stopwords')


# In[39]:


# printing English Stopwords

print(stopwords.words('english'))


# In[4]:


# loading dataset from local directory

dataset = pd.read_csv('C:/Users/shubhangi/Desktop/news.csv')


# In[5]:


# printing first 4 rows of dataset

dataset.head(4)


# In[6]:


## Checking number of rows and columns of dataset

dataset.shape


# In[7]:


# Labelling 'Real' News As '0' & 'Fake' News As '1'

label = {"REAL": 0, "FAKE": 1}
dataset['label'] = dataset['label'].map(label)
dataset.head()


# In[8]:


## checking number of null values from dataset

dataset.isnull().sum()


# In[9]:


# Using Porter Stemming algorithm to remove commener morphological words and affixes of words with same meaning

stmg = PorterStemmer()


# In[10]:


def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]',' ',title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [stmg.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = ' '.join(stemmed_title)
    return stemmed_title


# In[11]:


dataset['title'] = dataset['title'].apply(stemming)


# In[12]:


print(dataset['title'])


# In[13]:


# Removing column label from dataset

X = dataset.drop(columns='label', axis=1)
Y = dataset['label']


# In[14]:


print(X)
print(Y)


# In[15]:


X = dataset['title'].values
Y = dataset['label'].values


# In[16]:


print(X)


# In[17]:


print(Y)


# In[18]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[34]:


print(X[:7])


# In[20]:


# Splitting data for training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y, random_state=2)


# In[21]:


model = LogisticRegression()


# In[22]:


model.fit(X_train, Y_train)


# In[23]:


# accuracy score on the training data

X_train_predict = model.predict(X_train)
training_accuracy = accuracy_score(X_train_predict, Y_train)


# In[24]:


print('Accuracy score of the training data : ', training_accuracy)


# In[25]:


# accuracy score on the test data
X_test_predict = model.predict(X_test)
test_accuracy = accuracy_score(X_test_predict, Y_test)


# In[26]:


print('Accuracy score of the test data : ', test_accuracy)


# In[27]:


X_new = X_test[2]

pred = model.predict(X_new)
print(pred)

if (pred[0]==0):
    print('The news is Real')
else:
    print('The news is Fake')


# In[29]:


print(Y_test[2])


# In[ ]:




