# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:42:51 2016

@author: juanpablo
"""
#Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
import shapely.geometry as geom
import geopandas as gpd
import sklearn as sklearn
from sklearn import cluster
from sklearn import preprocessing
from matplotlib.colors import ListedColormap

#Importing data and creating lat long list
food = pd.read_csv('food_final.csv')
food = food[food.SCORE > -2]
latlon = pd.DataFrame(list(food.latlon.str.split()))
latlon.columns = ['latitude', 'longitude']
latlon['latitude'] = latlon['latitude'].map(lambda x: str(x)[:-1])
latlon = latlon.astype(float)
latlon = latlon[['longitude', 'latitude']]
boros = gpd.GeoDataFrame.from_file('nycb.shp')
#points = map(geom.Point, latlon.as_matrix().tolist())
#gpos = gpd.GeoSeries(points)
#boros.plot(figsize=(10,8), alpha=0.3)

#Plotting geographical distribution of the points
plt.figure(figsize=(10,8))
boros.plot(figsize=(10,8), alpha=0)
plt.scatter(latlon.longitude, latlon.latitude, s=20, alpha=0.3)
plt.title("Distribution of food stablishments @ NYC", fontsize = 20)
plt.ylim([40.45, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

#Plotting DOH Score Vs Ratings and price tier Vs score
ax = food.plot(x='rating', y='SCORE', kind='scatter', figsize=(8,6), grid=True,
          title="DOH Score Vs 4Square ratings", alpha=0.5) 
ax.set(xlabel="Rating", ylabel="DOH Score")          
print(spst.pearsonr(food.rating, food.SCORE))
ax1 = food.plot(x='price.tier', y='SCORE', kind='scatter', figsize=(8,6), 
                grid=True, title="DOH Score Vs 4Square price range", alpha=0.5)
ax1.set(xlabel="Price Tier", ylabel="DOH Score")

#Subsetting and whitening the data for clustering
food_sel = food[['rating','SCORE','price.tier', 'stats.checkinsCount']]
food_sel['price.tier'] = food_sel['price.tier'].fillna(value=1)
food_sel1 = food_sel[['rating', 'SCORE']]
food_sel1_scaled = preprocessing.scale(food_sel1)
food_sel_scaled = preprocessing.scale(food_sel)

#Clustering with kmeans and only rating and DOH Score
k = 3
est = cluster.KMeans(n_clusters = k, n_init = 100)
est.fit(food_sel1_scaled)
colors = est.labels_.astype(np.float) 
cmap = ListedColormap(['blue','green','yellow','red', 'cyan'])
plt.figure(figsize=(10, 8))
boros.plot(figsize=(10,8), alpha=0)
plt.scatter(latlon.longitude, latlon.latitude, s=40, c=colors, cmap=cmap, 
            alpha = 0.5)
plt.title("Clustering with K-means and 3 clusters", fontsize = 20)
plt.xlabel('Latitude', fontsize=16)
plt.ylabel('Longitude', fontsize=16)
plt.ylim([40.5, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

#Clustering with kmeans and all featrures (rating, DOH Score, price tier and
#4squared checkins)
k = 2
est = cluster.KMeans(n_clusters = k, n_init = 100)
est.fit(food_sel_scaled)
colors = est.labels_.astype(np.float)
cmap = ListedColormap(['blue','green','yellow','red', 'cyan'])
plt.figure(figsize=(10, 8))
boros.plot(figsize=(10,8), alpha=0)
plt.scatter(latlon.longitude, latlon.latitude, s=40, c=colors, cmap=cmap, 
            alpha =0.5)
plt.title("Clustering with K-means and 2 clusters", fontsize = 20)
plt.xlabel('Latitude', fontsize=16)
plt.ylabel('Longitude', fontsize=16)
plt.ylim([40.5, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()
#db = cluster.DBSCAN(eps=0.7, min_samples=8).fit(food_sel)
#labels = db.labels_
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#Dividing the sample into train and test sets
np.random.seed(2015)
ind=spst.bernoulli.rvs(p = 0.6, size = len(food_sel))
food_sel_train = food_sel[ind==1]
food_sel_test = food_sel[ind==0]          