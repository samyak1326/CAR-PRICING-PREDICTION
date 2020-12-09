#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for Data visuaization
import matplotlib.pyplot as plt # for Data visuaization
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# data.head()

# # data = pd.read_csv('Automobile_dataset.csv')
# data.head()

# In[3]:


data.info()


# In[4]:


#Replacing ? with NaN
data = data.replace('?',np.NaN)
data


# In[5]:


#Identifying where NaN is present and how many
data.isna().sum()


# In[6]:


data['num-of-doors'].fillna(data['num-of-doors'].mode()[0], inplace=True)
data['normalized-losses'].fillna(data['normalized-losses'].mode()[0], inplace=True)
data['bore'].fillna(data['bore'].mode()[0], inplace=True)
data['stroke'].fillna(data['stroke'].mode()[0], inplace=True)
data['horsepower'].fillna(data['horsepower'].mode()[0], inplace=True)
data['peak-rpm'].fillna(data['peak-rpm'].mode()[0], inplace=True)
data['price'].fillna(data['price'].mode()[0], inplace=True)
data.isna().sum()


# In[7]:


#Price Distribution plot
plt.figure(figsize=(10,10),frameon=False)
plt.tight_layout()
plt.title('FREQUENCY')
sns.distplot(data['price'])
plt.ylabel('Frequency')
#Distribution shows that maximum car are less than 200000


# In[8]:


# Maker Distribution Plot
plt.figure(figsize=(25,10),frameon=False)
sns.countplot(x='make',data=data)
plt.title('CAR MAKER DISTRIBUTION')
#checked_1 # Toyota : fovourite


# In[9]:


data


# In[10]:


# Summary of the Dataset: 205 rows, 26 columns, no null values
print(data.info())


# In[11]:


# BORE, STROKE, HP, P-RPM, PRICE
data['bore'] = data['bore'].astype(float)
data['stroke'] = data['stroke'].astype(float)
data['horsepower'] = data['horsepower'].astype(int)
data['peak-rpm'] = data['peak-rpm'].astype(int)
data['price'] = data['price'].astype(float)
print(data.info())


# In[12]:


data["avgmpg"] = (data["city-mpg"]+data["highway-mpg"])/2
data["avgmpg"].head(10)


# In[13]:


df_cars = data[['price', 'make', 'fuel-type', 'aspiration','body-style', 'wheel-base', 'engine-type', 'num-of-cylinders', 'fuel-system', 'drive-wheels','engine-size', 'curb-weight', 'length', 'width', 'bore', 'stroke', 'horsepower', 'avgmpg' ]]
df_cars


# In[14]:


df_cars.head()


# In[15]:


df_dummy = pd.get_dummies(df_cars['make'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('make', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['fuel-type'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('fuel-type', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['aspiration'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('aspiration', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['body-style'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('body-style', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['drive-wheels'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('drive-wheels', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['engine-type'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('engine-type', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['num-of-cylinders'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('num-of-cylinders', axis = 1, inplace=True)

df_dummy = pd.get_dummies(df_cars['fuel-system'])
df_cars = pd.concat([df_cars, df_dummy], axis = 1)
df_cars.drop('fuel-system', axis = 1, inplace=True)

print(df_cars.shape)


# In[23]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
y = df_cars['price']
X = df_cars.drop(['price'], axis = 1)
X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
#.fit_transform first fits the original data and then transforms it
X_test = scaler.transform(X_test_org)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics

lreg = LinearRegression()
lreg.fit(X_train, y_train)

print("R2 Training Score: ", lreg.score(X_train, y_train))
print("R2 Testing Score: ", lreg.score(X_test, y_test))


# In[25]:


print(lreg.intercept_)
lreg.coef_


# In[30]:


test_predict = lreg.predict(X_test)
test_predict = pd.DataFrame(test_predict,columns=['Predicted_Price'])
test_predict['Predicted_Price'] = round(test_predict['Predicted_Price'],2)
y_test_index = y_test.reset_index()
y_test_index = y_test_index.drop(columns='index', axis = 1)
test_predict = pd.concat([y_test_index, test_predict], axis = 1)
test_predict.head(30)


# In[32]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(x="price", y="Predicted_Price", data=test_predict)
plt.subplot(1,2,2)
sns.regplot(x="price", y="Predicted_Price", data=test_predict)


# In[ ]:





# In[ ]:




