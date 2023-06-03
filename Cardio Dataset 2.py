#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('Cardio Desease Final Data.csv')


# In[3]:


data.head()


# In[4]:


data=data.drop('Unnamed: 0',axis=1)


# In[5]:


data.head()


# In[6]:


data['height']=data['height']/100


# In[7]:


data['BMI']=(data['weight'])/(data['height']**2)


# In[8]:


data=data.drop(columns=['height','weight'],axis=1)


# In[9]:


data.BMI=data.BMI.round(2)


# In[10]:


data


# In[11]:


x=data.drop('cardio',axis=1)


# In[12]:


y=data['cardio']


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


# In[16]:


lr=LogisticRegression()
mnb=MultinomialNB()
svc=SVC()
rfc=RandomForestClassifier()
dtc=DecisionTreeClassifier()


# In[17]:


lr.fit(x_train,y_train)


# In[18]:


lr.score(x_test,y_test)


# In[19]:


mnb.fit(x_train,y_train)


# In[20]:


mnb.score(x_test,y_test)


# In[21]:


rfc.fit(x_train,y_train)


# In[22]:


rfc.score(x_test,y_test)


# In[23]:


dtc.fit(x_train,y_train)


# In[24]:


dtc.score(x_test,y_test)


# In[25]:


svc.fit(x_train,y_train)


# In[ ]:


svc.score(x_test,y_test)


# In[ ]:


data.columns=data.columns.str.upper()


# In[ ]:


data

