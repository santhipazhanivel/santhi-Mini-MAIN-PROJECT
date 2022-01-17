#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("C:\\Users\\home\\Desktop\\ml project\\diabetes.csv")
data.head(12)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data.tail(5)


# In[7]:


data.head(5)


# In[8]:


data.info()


# In[9]:


data.value_counts


# In[10]:


data.Pregnancies.unique()


# In[11]:


data["Pregnancies"].value_counts()


# In[12]:


data["Glucose"].value_counts()


# In[13]:


data["Glucose"].mean()


# In[14]:


data.corr()


# In[15]:


data.corr(method = "pearson")


# In[16]:


data.count()


# In[17]:


data.min()


# In[18]:


data.max()


# In[19]:


data.median()


# In[20]:


data.std()


# In[21]:


data["Pregnancies"]


# In[22]:


data[["Pregnancies", "Glucose", "BMI"]]


# In[23]:


data.Pregnancies.iloc[100]                          


# In[24]:


data.Pregnancies.loc[100]                          


# In[25]:


data.iloc[10,1]   


# In[26]:


data.head()


# In[27]:


data.isnull()


# In[28]:


data.notnull()


# In[29]:


data.dropna()


# In[30]:


data['BMI'].fillna("119",inplace=True)
data["BMI"].isnull().value_counts()


# In[31]:




from sklearn.model_selection import train_test_split


# In[33]:


x=data.iloc[:,data.columns!='Outcome']#data
y=data.iloc[:,data.columns=='Outcome']#outcome


# In[34]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


# In[35]:


xtrain.head()


# In[36]:




ytest.head()


# In[37]:


xtrain.head()


# In[38]:


xtest.head()


# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


model=RandomForestClassifier()


# In[41]:


model.fit(xtrain,ytrain.values.ravel())


# In[42]:


predict_output = model.predict(xtest)# to predict ouput acording to the previeus output(x test)
print(predict_output)


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


acc=accuracy_score(predict_output,ytest)
print("the accuracy score for RF",acc)


# In[ ]:





# In[ ]:




