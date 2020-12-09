#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from bokeh.io import output_notebook
from bokeh.plotting import ColumnDataSource, figure, show
from sklearn import metrics


# In[2]:


df = pd.read_csv('Automobile_dataset.csv')
df.shape


# In[3]:


df.describe()


# In[4]:


df.head()


# In[5]:


#drop symbolising, losses and eng_location 
#ez1
df.drop(['symboling','normalized-losses','engine-location'], axis=1, inplace=True)
df = df.replace('?',np.NaN)
df


# In[6]:


df.isna().sum()


# In[7]:


#value made 0
df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)  
df['bore'].fillna(df['bore'].mode()[0], inplace=True)
df['stroke'].fillna(df['stroke'].mode()[0], inplace=True) 
df['horsepower'].fillna(df['horsepower'].mode()[0], inplace=True)
df['peak-rpm'].fillna(df['peak-rpm'].mode()[0], inplace=True)
df['price'].fillna(df['price'].mode()[0], inplace=True)

df.isna().sum()


# In[8]:


#plot 1
#Price Distribution plot
plt.figure(figsize=(10,10),frameon=False)
plt.tight_layout()
sns.distplot(df['price'])
plt.ylabel('Frequency')
#This plot distribution shows that maximum cars are less than 200000


# In[9]:


# Car Maker Distribution Plot
plt.figure(figsize=(25,10),frameon=False)
sns.countplot(x='make',data=df)
plt.title('CAR MAKER DISTRIBUTION')
#checked_2 #TOYOTA


# In[10]:


# horsepower
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
#price
df['price'] = pd.to_numeric(df['price'], errors='coerce')
# enable notebook output
output_notebook()
source = ColumnDataSource(data=dict(
 x=df['horsepower'],
 y=df['price'],
 make=df['make'],
))
tooltips = [
 ('make', '@make'),
 ('horsepower', '$x'),
 ('price', '$y{$0}')
]
s = figure(plot_width=600, plot_height=400, tooltips=tooltips)
s.xaxis.axis_label = 'Horsepower'
s.yaxis.axis_label = 'Price'

# Add a square renderer with a size, color, and alpha
s.circle('x', 'y', source=source, size=8, color='blue', alpha=0.5)

# Displaying the Result
show(s)


# In[11]:


#plot : fuel type vs city-mpg
sns.stripplot(df['fuel-type'], df['city-mpg'])


# In[12]:


sns.pairplot(df[['city-mpg', 'wheel-base', 'engine-size']])


# In[13]:


df


# In[14]:


print(df.info())


# In[15]:


# all numeric (int and float) variables in the dataset
cars_numeric = df.select_dtypes(include=['float64', 'int64'])
cars_numeric.head()


# In[16]:


#correlation plot
correlation=cars_numeric.corr()
correlation


# In[19]:


# Method to plot correlation using a HEAT MAP
# utube

# Figure size
plt.figure(figsize=(17,8))

# heatmap
sns.heatmap(correlation, cmap="YlGnBu", annot=True)
plt.show()


# In[20]:


# 0-204
cars_numeric


# In[21]:


#price
x = cars_numeric.iloc[:,0:10].values
y = cars_numeric.iloc[:,10].values
y


# In[22]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
x_train.shape


# In[23]:


x_test.shape


# In[24]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[25]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred


# In[26]:


y_test


# In[27]:


m = model.intercept_
m


# In[28]:


n = model.coef_
n


# In[29]:


model.predict([x_train[10]])


# In[30]:


y_train[10]


# In[31]:


#scatter plot
plt.scatter(y_pred,y_test)


# In[32]:


df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df1


# In[33]:


df2 = df1
df2.plot(figsize=(20,8),kind='bar')
plt.show()


# In[34]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(x="Actual", y="Predicted", data=df1)
plt.subplot(1,2,2)
sns.regplot(x="Actual", y="Predicted", data=df1)


# In[35]:


# metrics
# Mean Squared Error
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_true=y_test, y_pred=y_pred))


# In[36]:


MSE = np.square(np.subtract(y_test,y_pred)).mean()
MSE


# In[37]:


# Error terms
c = [i for i in range(len(y_pred))]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")

# Plot heading
fig.suptitle('Error', fontsize=20)

# X-label
plt.xlabel('Index', fontsize=18)

# Y-label
plt.ylabel('ytest-ypred', fontsize=16)

# Display
plt.show()

#CHECK FOR ERROR (IMPORTANT)


# In[ ]:




