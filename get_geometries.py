
# coding: utf-8

# In[ ]:

import numpy as np
import pysal as ps
get_ipython().magic(u'matplotlib inline')
import pandas as pd
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
import random
from shapely.geometry import Polygon, Point


# In[ ]:

#read in file 
food = pd.read_csv('food_final.csv')
food.head()


# In[ ]:

#split latlon column into two new columns 
food['lat'], food['lon'] = zip(*food['latlon'].apply(lambda x: x.split(',')))


# In[ ]:

food.head()


# In[ ]:

#make sure that they are float values
food[['lon']] = food[['lon']].astype(float)
food[['lat']] = food[['lat']].astype(float)


# In[ ]:

#using the shapely module, create geometries from the coordinates
food['geometry'] = food.apply(lambda row: Point(row['lon'], row['lat']), axis=1)


# In[ ]:

#inspect data
food['geometry'].head()


# In[ ]:

#food.to_csv('food_geo.csv')


# In[ ]:



