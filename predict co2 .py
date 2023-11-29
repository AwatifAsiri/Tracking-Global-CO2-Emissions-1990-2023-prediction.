#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=pd.read_csv("C:/Users/hp/Downloads/tidy_format_co2_emission_dataset.csv")


# In[7]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


import os
df['CO2EmissionRate (mt)']=df['CO2EmissionRate (mt)'].replace('-',float('nan'))
df['CO2EmissionRate (mt)'].fillna(method='ffill',inplace=True)


# In[10]:


df.loc[df['CO2EmissionRate (mt)']=='-','CO2EmissionRate (mt)']=df.iloc[df[df['CO2EmissionRate (mt)']=='-'].index+1,2]


# In[12]:


df['CO2EmissionRate (mt)']=df['CO2EmissionRate (mt)'].apply(lambda x: float(str(x).replace(',','')))


# In[13]:


df['CO2EmissionRate (mt)']=df['CO2EmissionRate (mt)'].astype('float')


# In[14]:


df.info()


# In[20]:


df.isnull()


# In[21]:


trend=df.groupby('Year')['CO2EmissionRate (mt)'].sum()[:-1]
trend


# In[34]:


df['CO2EmissionRate (mt)'].mean()


# In[35]:


df['CO2EmissionRate (mt)'].max()


# In[39]:


df['Country'].nunique()


# In[38]:


df['Country'].value_counts().head(5)


# In[40]:


df2= df.groupby('Country')['CO2EmissionRate (mt)'].mean().sort_values(ascending=False)


# In[41]:


df2


# In[42]:


sns.pairplot(df)


# In[28]:


plt.figure(figsize=(11,5))
sns.lineplot(x=trend.index,y=trend.values)
plt.xticks(rotation=90)
plt.title('Global CO2 Emission tredn from 1990 to 2018')
plt.xlabel('Year')
plt.ylabel('Co2 Emission in mt')
plt.show()


# # Training a linear regression model

# In[15]:


df.columns


# In[16]:


X =df[['Country', 'Year']]
y =df['CO2EmissionRate (mt)']



# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


lm = LinearRegression()


# In[21]:


lm.fit(X_train,y_train)


# In[22]:


from sklearn.linear_model import LinearRegression
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')


# In[43]:


X.fillna(0, inplace=True)
y.fillna(0, inplace=True)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[25]:


lm.fit(X_train,y_train)


# In[26]:


# print the intercept
print(lm.intercept_)


# In[27]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# # Predictions Model

# In[28]:


predictions = lm.predict(X_test)


# In[29]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[30]:


sns.distplot((y_test-predictions),bins=50);


# In[31]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




