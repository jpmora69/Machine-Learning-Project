
# coding: utf-8

# In[ ]:

import numpy as np
import pysal as ps
import pandas as pd
import geopandas as gp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pandas.stats.api import ols
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
pd.set_option('display.max_rows', 500)
get_ipython().magic(u'matplotlib inline')


# In[ ]:

data = pd.read_csv('food_geo3.csv')
data.head()


# In[ ]:

#setting the weight matrix for spatial autocorrelation
#using k-nearest neighbors-3,5,and9
w_knn3 = ps.knnW_from_shapefile(('newer_polys.shp'), k=3, idVariable='CAMIS')
w_knn5 = ps.knnW_from_shapefile(('newer_polys.shp'), k=5, idVariable='CAMIS')
w_knn9 = ps.knnW_from_shapefile(('newer_polys.shp'), k=9, idVariable='CAMIS')


# In[ ]:

#normalize the dependent variable
Y = data.SCORE.values
Y = (Y-Y.mean())/Y.std() 
print Y.shape


# In[ ]:

# Now we would like to standardize all the weights. This can be 
# done by specifying 'R' as the matrix transformation.
w_knn3.transform = 'R'
w_knn5.transform = 'R'
w_knn9.transform = 'R'


# In[ ]:

# and then compute the spatial lag for all neighborhoods based
# on the spatial weight matrix. We also store this as a column
# named 'w_percent_knn3' in the original table.
sl = ps.lag_spatial(w_knn3, Y)
data['w_percent_knn3'] = sl


# In[ ]:

data.head()


# In[ ]:

#calculate moran's i 
moran = ps.Moran(Y, w_knn3)


# In[ ]:

#show moran's i
moran.I


# In[ ]:

#the p-value
moran.p_sim


# In[ ]:

# It's time to look at the Moran Scatter Plot to inspet the results
f, ax = plt.subplots(1, figsize=(10,10))
sns.regplot(x='SCORE', y='w_percent_knn3', data=data)
plt.axvline(0, c='k', alpha=0.5)
plt.axhline(0, c='k', alpha=0.5)
plt.title("Moran's I value - inspection grades and businesses")
plt.show()
f.savefig('moranI.png')


# In[ ]:

#local moran's i 
lisa= ps.Moran_Local(Y, w_knn3)


# In[ ]:

# Let's narrow down to those neighborhoods that are
# statistically significant.

S = lisa.p_sim < 0.05


# In[ ]:

#And which quadrants they belong to
#1 = HighHigh, 2 = LowHigh, 3 = LowLow, 4 = HighLow
#1 means that business has a high SCORE (bad--higher SCORE means bad grade) and is around other businesses with a high SCORE
#2 means that business has a low SCORE and is around around other businesses with a high SCORE
#3 means that a business has a low SCORE and is around other businesses with a low SCORE
#4 means that a business has a high SCORE and is around other businesses with low score
Q = lisa.q


# In[ ]:

# Next, we'll turn those into a GeoDataFrame for visualization.

records = map(lambda x: (data.iloc[x]['CAMIS'],data.iloc[x]['DBA'], Q[x],data.iloc[x]['SCORE'],
                         data.geometry.iloc[x], data.lat[x], data.lon[x]),
              [i for i,s in enumerate(S) if s])


gdata = gp.GeoDataFrame(records, columns=('CAMIS','Name', 'quadrant','SCORE', 'geometry',
                                         'lat', 'lon'))
gdata.head()


# In[ ]:

# and then compute the spatial lag for all neighborhoods based
# on the spatial weight matrix. We also store this as a column
# named 'w_percent_knn5' in the original table.
s2 = ps.lag_spatial(w_knn5, Y)
data['w_percent_knn5'] = s2
data.head()


# In[ ]:

#calculating moran's i 
moran2 = ps.Moran(Y, w_knn5)


# In[ ]:

#showing moran's i
moran2.I


# In[ ]:

#the p value
moran2.p_sim


# In[ ]:

#local moran's i
lisa2 = ps.Moran_Local(Y, w_knn5)


# In[ ]:

# Let's narrow down to those neighborhoods that are
# statistically significant.

S = lisa2.p_sim < 0.05


# In[ ]:

#And which quadrants they belong to
#1 = HighHigh, 2 = LowHigh, 3 = LowLow, 4 = HighLow
#1 means that business has a high SCORE (bad--higher SCORE means bad grade) and is around other businesses with a high SCORE
#2 means that business has a low SCORE and is around around other businesses with a high SCORE
#3 means that a business has a low SCORE and is around other businesses with a low SCORE
#4 means that a business has a high SCORE and is around other businesses with low score
Q = lisa2.q


# In[ ]:

# Next, we'll turn those into a GeoDataFrame for visualization.

records = map(lambda x: (data.iloc[x]['CAMIS'],data.iloc[x]['DBA'], Q[x],data.iloc[x]['SCORE'],
                         data.geometry.iloc[x]),
              [i for i,s in enumerate(S) if s])


gdata2 = gp.GeoDataFrame(records, columns=('CAMIS','Name', 'quadrant','SCORE', 'geometry'))
gdata2.head()


# In[ ]:

# It's time to look at the Moran Scatter Plot to inspet the results
f, ax = plt.subplots(1, figsize=(10,10))
sns.regplot(x='SCORE', y='w_percent_knn5', data=data)
plt.axvline(0, c='k', alpha=0.5)
plt.axhline(0, c='k', alpha=0.5)
plt.title("Moran's I value - inspection grades and businesses")
plt.show()
f.savefig('moranI2.png')


# In[ ]:

# and then compute the spatial lag for all neighborhoods based
# on the spatial weight matrix. We also store this as a column
# named 'w_percent_knn5' in the original table.
s3 = ps.lag_spatial(w_knn9, Y)
data['w_percent_knn9'] = s3
data.head()


# In[ ]:

#calculate moran's i 
moran3 = ps.Moran(Y, w_knn9)


# In[ ]:

#showing moran's i 
moran3.I


# In[ ]:

#the pa value
moran3.p_sim


# In[ ]:

#local moran's i 
lisa3 = ps.Moran_Local(Y, w_knn9)


# In[ ]:

# Let's narrow down to those neighborhoods that are
# statistically significant.

S = lisa3.p_sim < 0.05


# In[ ]:

#And which quadrants they belong to
#1 = HighHigh, 2 = LowHigh, 3 = LowLow, 4 = HighLow
#1 means that business has a high SCORE (bad--higher SCORE means bad grade) and is around other businesses with a high SCORE
#2 means that business has a low SCORE and is around around other businesses with a high SCORE
#3 means that a business has a low SCORE and is around other businesses with a low SCORE
#4 means that a business has a high SCORE and is around other businesses with low score
Q = lisa3.q


# In[ ]:

# Next, we'll turn those into a GeoDataFrame for visualization.

records = map(lambda x: (data.iloc[x]['CAMIS'],data.iloc[x]['DBA'], Q[x],data.iloc[x]['SCORE'],
                         data.geometry.iloc[x]),
              [i for i,s in enumerate(S) if s])


gdata3 = gp.GeoDataFrame(records, columns=('CAMIS','Name', 'quadrant','SCORE', 'geometry'))
gdata3.head()


# In[ ]:

# It's time to look at the Moran Scatter Plot to inspet the results
f, ax = plt.subplots(1, figsize=(10,10))
sns.regplot(x='SCORE', y='w_percent_knn3', data=data)
plt.axvline(0, c='k', alpha=0.5)
plt.axhline(0, c='k', alpha=0.5)
plt.title("Moran's I value - inspection grades and businesses")
plt.show()
f.savefig('moranI3.png')


# In[ ]:



