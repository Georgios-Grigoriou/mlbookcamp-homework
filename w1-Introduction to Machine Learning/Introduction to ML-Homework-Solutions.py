#!/usr/bin/env python
# coding: utf-8

# ## Homework
# 
# ### Set up the environment
# 
# You need to install Python, NumPy, Pandas, Matplotlib and Seaborn. For that, you can the instructions from
# [06-environment.md](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/01-intro/06-environment.md).

# #### Import libraries

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Question 1
# 
# What's the version of Pandas that you installed?
# 
# You can get the version information using the `__version__` field:

# In[2]:


pd.__version__


# ### Getting the data 
# 
# For this homework, we'll use the California Housing Prices dataset. Download it from 
# [here](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv).
# 
# You can do it with wget:
# 
# ```bash
# wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
# ```
# 
# Or just open it with your browser and click "Save as...".
# 
# Now read it with Pandas.

# In[3]:


df=pd.read_csv(r'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# ### Question 2
# 
# How many columns are in the dataset?

# In[6]:


columns=df.columns
print("The Columns in the data set are {}".format(len(columns)))
print("The Columns are:")
print("-----------------")
print(pd.Series(columns))


# ### Question 3
# 
# Which columns in the dataset have missing values?

# In[7]:


# Find the columns that have missing values
df.isna().any()


# In[8]:


# Find the Columns and how many are the missing values
df.isna().sum()


# **The column total_bedrooms has missing values**

# ### Question 4
# 
# How many unique values does the `ocean_proximity` column have?
# 
# - 3
# - 5
# - 7
# - 9

# In[9]:


df.ocean_proximity.unique()


# In[10]:


len(df.ocean_proximity.unique())


# **Unique values in the ocean_proximity column are 5**

# ### Question 5
# 
# What's the average value of the `median_house_value` for the houses located near the bay?
# 

# In[11]:


# find the values where ocean proximity is near the bay
near_the_bay=(df.ocean_proximity=='NEAR BAY')


# In[12]:


df.median_house_value[near_the_bay].mean()


# **The correct answer is 259212**

# ### Question 6
# 
# 1. Calculate the average of `total_bedrooms` column in the dataset.
# 2. Use the `fillna` method to fill the missing values in `total_bedrooms` with the mean value from the previous step.
# 3. Now, calculate the average of `total_bedrooms` again.
# 4. Has it changed?
# 
# Has it changed?
# 
# > Hint: take into account only 3 digits after the decimal point.

# In[13]:


mean_total_bedrooms=df.total_bedrooms.mean()
mean_total_bedrooms


# In[14]:


df[df.total_bedrooms.isna()]


# In[15]:


df.total_bedrooms.fillna(mean_total_bedrooms.mean(),inplace=True)


# In[16]:


df[df.total_bedrooms.isna()]


# In[17]:


mean_total_bedrooms_new=df.total_bedrooms.mean()


# In[18]:


abs(mean_total_bedrooms_new-mean_total_bedrooms)


# In[19]:


abs(mean_total_bedrooms_new-mean_total_bedrooms)/(mean_total_bedrooms)*100


# **The correct answer is NO, if we consider 3 digits after the decimal point. The reason is that we have 207 missing values out of 20640 which account for 1% of the total dataset thus the minor change**

# ### Question 7
# 
# 1. Select all the options located on islands.
# 2. Select only columns `housing_median_age`, `total_rooms`, `total_bedrooms`.
# 3. Get the underlying NumPy array. Let's call it `X`.
# 4. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
# 5. Compute the inverse of `XTX`.
# 6. Create an array `y` with values `[950, 1300, 800, 1000, 1300]`.
# 7. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
# 8. What's the value of the last element of `w`?

# 1. Select all the options located on islands.

# In[20]:


df_islands=df[df.ocean_proximity=='ISLAND']
df_islands


# 2. Select only columns housing_median_age, total_rooms, total_bedrooms.

# In[21]:


df_islands=df_islands[['housing_median_age','total_rooms','total_bedrooms']]
df_islands


# 3. Get the underlying NumPy array. Let's call it X.

# In[22]:


X=np.array(df_islands)


# In[23]:


X


# 4. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.

# In[24]:


X_transpose=X.T


# In[25]:


X_transpose


# In[26]:


XTX=np.dot(X_transpose,X)


# In[27]:


XTX


# 5. Compute the inverse of XTX.

# In[28]:


XTX_inverse=np.linalg.inv(XTX)


# In[29]:


XTX_inverse


# 6. Create an array `y` with values `[950, 1300, 800, 1000, 1300]`.
# 

# In[30]:


y=np.array([950, 1300, 800, 1000, 1300])


# 7. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
# 

# In[31]:


Xdot=np.dot(XTX_inverse,X_transpose)


# In[32]:


w=np.dot(Xdot,y)


# 8. What's the value of the last element of `w`?

# In[33]:


w


# **The correct answer is 5.6992**
