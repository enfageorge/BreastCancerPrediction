#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data= pd.read_csv('data.csv')


# In[3]:


data


# # Data Preprocessing

# In[4]:


# Missing values
data.isnull().sum()


# In[5]:


data.isna().sum() # No missing values


# # Data Exploration

# In[6]:


data.head()


# In[7]:


data.shape


# # Data Preparation

# In[8]:


X = data.iloc[:,2:32].values
Y = data.iloc[:,1].values


# In[9]:


Y


# In[10]:


#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder #encode categorical features using a one-hot or ordinal encoding scheme
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[11]:


Y


# In[12]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[13]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# # Prediction 

# In[15]:


Y_pred = classifier.predict(X_test)


# In[16]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[17]:


cm


# In[18]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)


# In[19]:


accuracy


# In[ ]:




