
# coding: utf-8

# In[19]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(df)
    inertias.append(model.inertia_)
plt.plot(ks,inertias,'-o')
plt.xlabel('no of cluster,k')
plt.ylabel('Inertias')
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('C:/Users/snaras01/Downloads/data_stocks.csv')

#=========Determining the number of cluster using Elbow method===========
print("Determining the number of cluster using Elbow method")

ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(df)
    inertias.append(model.inertia_)
plt.plot(ks,inertias,'-o')
plt.xlabel('no of cluster,k')
plt.ylabel('Inertias')

df_data = df.iloc[:,1:-1].values

kmeans = KMeans(n_clusters=4)

labels = kmeans.fit_predict(df_data)

df['labels'] = labels
df.sort_values('labels')

#===Number of unique patterns=====
print("Number of unique patterns")

df['labels'].unique()

#============stocks moving together and stocks are different from each other===
print("stocks moving together and stocks are different from each other")
print("stocks apparently similar in performance")
df_cat0= df.loc[df['labels']==0]
df_cat1= df.loc[df['labels']==1]
df_cat2=df.loc[df['labels']==2]
df_cat2=df.loc[df['labels']==3]






# In[3]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('C:/Users/snaras01/Downloads/data_stocks.csv')
df


# In[17]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
print("Determining the number of cluster using Elbow method")

ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(df)
    inertias.append(model.inertia_)
plt.plot(ks,inertias,'-o')
plt.xlabel('no of cluster,k')
plt.ylabel('Inertias')


# In[7]:

df_data = df.iloc[:,1:-1].values
df_data


# In[8]:

kmeans = KMeans(n_clusters=4)

labels = kmeans.fit_predict(df_data)

df['labels'] = labels
df.sort_values('labels')


# In[20]:

print("stocks moving together and stocks are different from each other")
print("stocks apparently similar in performance")
df_cat0= df.loc[df['labels']==0]
df_cat1= df.loc[df['labels']==1]
df_cat2=df.loc[df['labels']==2]
df_cat2=df.loc[df['labels']==3]
print(df_cat0)


# In[21]:

df_cat1= df.loc[df['labels']==1]
print(df_cat1)


# In[22]:

df_cat2=df.loc[df['labels']==2]
print(df_cat2)


# In[23]:

df_cat3=df.loc[df['labels']==3]
print(df_cat3)


# In[ ]:



