
# coding: utf-8

# In[13]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv("C:\Machine Learning\stocks\data_stocks.csv")
print(df)




# In[15]:

# determining the number of cluster by elbow method
import matplotlib.pyplot as plt
ks = range(1,10)
inertias = []
for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(df)
    inertias.append(model.inertia_)
plt.plot(ks,inertias,'-o')
plt.xlabel('no of cluster,k')
plt.ylabel('Inertias')
plt.show()


# In[17]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv("C:\Machine Learning\stocks\data_stocks.csv")
df_data = df.iloc[:,1:-1].values

kmeans = KMeans(n_clusters=4)

labels = kmeans.fit_predict(df_data)

df['labels'] = labels
df.sort_values('labels')

#====Number of unique patterns========
df['labels'].unique()






# In[ ]:




# In[21]:

#====stocks moving together and stocks are different from each other===========
#====stocks apparently similar in performance=======================
df_cat0= df.loc[df['labels']==0]
df_cat1= df.loc[df['labels']==1]
df_cat2=df.loc[df['labels']==2]
df_cat2=df.loc[df['labels']==3]
print('*******************this data belongs to df_cat0*****************')
print(df_cat0)

print('*******************this data belongs to df_cat1*****************')
print(df_cat1)

print('*******************this data belongs to df_cat2*****************')
print(df_cat2)

print('*******************this data belongs to df_cat3*****************')
print(df_cat3)




# In[ ]:



