#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


data=pd.read_csv('cardio_train.csv',sep=';')
data


# In[3]:


data=data.drop('id',axis=1)


# In[4]:


data.isnull().sum()


# In[5]:


data['gender'].value_counts()


# In[6]:


data['cholesterol'].value_counts()


# In[7]:


data['gluc'].value_counts()


# In[8]:


data['smoke'].value_counts()


# In[9]:


data['alco'].value_counts()


# In[10]:


data['active'].value_counts()


# In[11]:


data['cardio'].value_counts()


# In[12]:


sb.countplot(data['cardio'])
plt.show()


# In[13]:


sb.countplot(x='active',data=data,hue='cardio')
plt.show()


# In[14]:


sb.countplot(x='cholesterol',data=data,hue='cardio')
plt.show()


# In[15]:


sb.countplot(x='gender',data=data,hue='cardio')
plt.show()


# In[16]:


corr=data.corr()


# In[17]:


plt.figure(figsize=(15,15))
sb.heatmap(corr,annot=True)
plt.show()


# In[18]:


age=data['age']/365


# In[19]:


age.max()


# In[20]:


age.min()


# In[21]:


plt.figure(figsize=(12,12))
data.boxplot()
plt.show()


# In[22]:


plt.boxplot(data['ap_hi'])


# In[23]:


data[data['ap_hi']>300].count().sum()


# In[24]:


plt.boxplot(data['ap_lo'])


# In[25]:


data[data['ap_lo']>10000]


# In[26]:


data[data['ap_lo']>200].count().sum()


# In[27]:


data['age']=data['age']/365


# In[28]:


data['age']=data['age'].round(2)


# In[29]:


data


# In[30]:


data.age.max()


# In[31]:


data.age.min()


# In[32]:


data.height.max()


# In[33]:


data[data['height']>200].count()


# In[34]:


data.height.min()


# In[35]:


data[data['height']<70].count()


# In[36]:


data.weight.max()


# In[37]:


data[data['weight']>150].count()


# In[38]:


data.weight.min()


# In[39]:


data[data['weight']<20].count()


# In[40]:


data[data['ap_hi']<data['ap_lo']]


# In[41]:


def fix_higher_value(value,threshold,divisor):
    if value>=threshold:
        return value/divisor
    else:
        return value


# In[42]:


data['ap_hi']=data['ap_hi'].apply(fix_higher_value,args=(10000,100))


# In[43]:


data['ap_hi']=data['ap_hi'].apply(fix_higher_value,args=(500,10))


# In[44]:


plt.boxplot(data['ap_hi'])


# In[45]:


data['ap_lo']=data['ap_lo'].apply(fix_higher_value,args=(10000,100))


# In[46]:


data['ap_lo']=data['ap_lo'].apply(fix_higher_value,args=(1000,10))


# In[47]:


data[data['ap_hi']<data['ap_lo']]


# In[48]:


data[data['ap_lo']>400].count().sum()


# In[49]:


data['ap_lo']=data['ap_lo'].apply(fix_higher_value,args=(500,10))


# In[50]:


def fix_lower_value(value,threshold,multiplyer):
    if value<=threshold:
        return value*multiplyer
    else:
        return value


# In[51]:


data[data['ap_hi']<0]


# In[52]:


data['ap_hi']=data['ap_hi'].apply(fix_lower_value,args=(0,-1))


# In[53]:


data[data['ap_hi']<0]


# In[54]:


data[data['ap_hi']<30]


# In[55]:


data['ap_hi']=data['ap_hi'].apply(fix_lower_value,args=(50,10))


# In[56]:


data[data['ap_hi']<data['ap_lo']]


# In[57]:


data['ap_lo']=data['ap_lo'].apply(fix_lower_value,args=(0,-1))


# In[58]:


mask=data['ap_hi']<data['ap_lo']


# In[59]:


data.loc[mask,['ap_hi','ap_lo']]=data.loc[mask,['ap_lo','ap_hi']].values


# In[60]:


data[data['ap_hi']<data['ap_lo']].count()


# In[61]:


data.shape


# In[62]:


data[data['ap_lo']==0].shape


# In[63]:


data['ap_hi'].nlargest(10)


# In[64]:


data['ap_hi'].nsmallest(10)


# In[65]:


data['ap_lo'].nlargest(10)


# In[66]:


data['ap_lo'].nsmallest(10)


# In[67]:


data.describe()


# In[68]:


data[(data['ap_lo']>0) & (data['ap_lo']<50)]


# In[69]:


data['ap_lo']=data['ap_lo'].apply(fix_lower_value,args=(10,10))


# In[70]:


data[(data['ap_lo']>0) & (data['ap_lo']<50)].shape


# In[71]:


data= data[(data['ap_hi']<300) & (data['ap_lo']>=45)]


# In[72]:


data


# In[73]:


corr_again=data.corr()


# In[74]:


plt.figure(figsize=(10,10))
sb.heatmap(corr_again,annot=True)
plt.show()


# In[75]:


x=data.drop('cardio',axis=1)


# In[76]:


y=data['cardio']


# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)


# In[79]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[80]:


lr=LogisticRegression()
mnb=MultinomialNB()
svc=SVC()
rfc=RandomForestClassifier()
dtc=DecisionTreeClassifier()


# In[81]:


lr.fit(x_train,y_train)


# In[82]:


lr.score(x_test,y_test)


# In[83]:


mnb.fit(x_train,y_train)


# In[84]:


mnb.predict(x_test)


# In[85]:


mnb.score(x_test,y_test)


# In[86]:


svc.fit(x_train,y_train)


# In[87]:


svc.score(x_test,y_test)


# In[88]:


rfc.fit(x_train,y_train)


# In[89]:


rfc.score(x_test,y_test)


# In[90]:


dtc.fit(x_train,y_train)


# In[91]:


dtc.score(x_test,y_test)


# # Try to improve the accuracy

# In[92]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[93]:


feature_select=SelectKBest(score_func=f_classif)


# In[94]:


feature_select.fit(x,y)


# In[95]:


score_col=pd.DataFrame(feature_select.scores_,columns=['Scores'])


# In[96]:


columns=pd.DataFrame(x.columns)


# In[97]:


scores=pd.concat([columns,score_col],axis=1)


# In[98]:


scores


# In[99]:


scores['Scores'].nlargest(7)


# In[100]:


data_modified=data.drop(columns=['gender','height','smoke','alco'],axis=1)


# In[101]:


data_modified


# In[102]:


X=data_modified.drop('cardio',axis=1)


# In[103]:


Y=data_modified['cardio']


# In[104]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[105]:


lr.fit(X_train,Y_train)


# In[106]:


lr.score(X_test,Y_test)


# In[ ]:





# In[107]:


Y_test


# In[108]:


mnb.fit(X_train,Y_train)


# In[109]:


mnb.score(X_test,Y_test)


# In[110]:


rfc.fit(X_train,Y_train)


# In[111]:


rfc.score(X_test,Y_test)


# In[112]:


dtc.fit(X_train,Y_train)


# In[113]:


dtc.score(X_test,Y_test)


# In[114]:


svc.fit(X_train,Y_train)


# In[115]:


svc.score(X_test,Y_test)


# In[116]:


data.to_csv('Cardio Desease Final Data.csv')

