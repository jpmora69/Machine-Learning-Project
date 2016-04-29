
# coding: utf-8
#by Juan Pablo

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
get_ipython().magic(u'matplotlib inline')
from rnn import RNN, MetaRNN

food = pd.read_csv('food_final.csv')
food = food[food.SCORE > 0]
latlon = pd.DataFrame(list(food.latlon.str.split()))
latlon.columns = ['latitude', 'longitude']
latlon['latitude'] = latlon['latitude'].map(lambda x: str(x)[:-1])
latlon = latlon.astype(float)
latlon = latlon[['longitude', 'latitude']]
boros = gpd.GeoDataFrame.from_file('nycb.shp')

#Plotting geographical distribution of the points
plt.figure(figsize=(10,8))
boros.plot(alpha=0)
plt.scatter(latlon.longitude, latlon.latitude, s=20, alpha=0.3)
plt.title("Distribution of food stablishments @ NYC", fontsize = 20)
plt.ylim([40.45, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

ax3 = food['SCORE'].plot(kind='hist', bins = 20, figsize=(6,4), grid=True, 
                title="Histogram of DOH Scores", alpha=0.8)
ax = food.plot(x='rating', y='SCORE', kind='scatter', figsize=(8,6), grid=True,
          title="DOH Score Vs 4Square ratings", alpha=0.5) 
ax.set(xlabel="Rating", ylabel="DOH Score")          
print(spst.pearsonr(food.rating, food.SCORE))
ax1 = food.plot(x='price.tier', y='SCORE', kind='scatter', figsize=(8,6), 
                grid=True, title="DOH Score Vs 4Square price range", alpha=0.5)
ax1.set(xlabel="Price Tier", ylabel="DOH Score")


ax2 = food['rating'].plot(kind='hist', bins = 20, figsize=(6,4), grid=True,
                 title="Histogram of 4Square ratings", alpha=0.8)


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
boros.plot(alpha=0)
plt.scatter(latlon.longitude, latlon.latitude, s=20, c=colors, cmap=cmap, 
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
boros.plot(alpha=0)
plt.scatter(latlon.longitude, latlon.latitude, s=20, c=colors, cmap=cmap, 
            alpha =0.5)
plt.title("Clustering with K-means and 2 clusters", fontsize = 20)
plt.xlabel('Latitude', fontsize=16)
plt.ylabel('Longitude', fontsize=16)
plt.ylim([40.5, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

food_sel_scaled_df = pd.DataFrame(preprocessing.scale(food_sel), columns =['rating','SCORE','price.tier', 'stats.checkinsCount'])

#Dividing the sample into train and test sets
np.random.seed(2015)
ind=spst.bernoulli.rvs(p = 0.75, size = len(food_sel1))
food_sel_scaled_train = food_sel_scaled_df[ind==1]
food_sel_scaled_test = food_sel_scaled_df[ind==0] 

#Creating parameters for training RNN
n_hidden = 5 # M
n_in = 3      # D
n_out = 1     # K
n_steps = 1  # the length of each sequence
n_seq = len(food_sel_scaled_train)   # the number of datapoints (i.e. sequences)

#Creating input and output arrays for training RNN         
rating = np.array(food_sel_scaled_train[['rating','price.tier','stats.checkinsCount']]).reshape(n_seq,n_steps,n_in)
score = np.array(food_sel_scaled_train['SCORE']).reshape(n_seq,n_steps,n_out)


#Creating the model and feeding it with training data set
model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=50, activation='tanh')
model.fit(rating, score, validation_frequency=5000)

guess = model.predict(rating.reshape(len(rating), n_in))
scores_pred = pd.DataFrame(guess)

scores_pred.columns = ['predictions']

scores_pred.predictions.plot(kind='hist', bins=20, figsize=(6,4), grid=True,                             title = "Histogram of predicted DOH normalized Scores (IS)", alpha=0.8)

food_sel_scaled_train.SCORE.plot(kind='hist', bins=20, figsize=(6,4), grid=True,                             title = "Histogram of actual DOH normalized Scores (IS)", alpha=0.8)

#Creating parameters for training RNN
n_hidden = 5 # M
n_in = 3      # D
n_out = 1     # K
n_steps = 1  # the length of each sequence
n_seq = len(food_sel_scaled_test)   # the number of datapoints (i.e. sequences)

#Creating input and output arrays for training RNN         
rating1 = np.array(food_sel_scaled_test[['rating','price.tier','stats.checkinsCount']]).reshape(n_seq,n_steps,n_in)
score1 = np.array(food_sel_scaled_test['SCORE']).reshape(n_seq,n_steps,n_out)

guess1 = model.predict(rating1.reshape(len(rating1), n_in))
scores_pred1 = pd.DataFrame(guess1)

scores_pred1.columns = ['predictions']

scores_pred1.predictions.plot(kind='hist', bins=20, figsize=(6,4), grid=True,                             title = "Histogram of predicted DOH normalized Scores (OS)", alpha=0.8)

food_sel_scaled_test.SCORE.plot(kind='hist', bins=20, figsize=(6,4), grid=True,                             title = "Histogram of actual DOH normalized Scores (OS)", alpha=0.8)

fig = plt.figure(figsize=(8,6))
ax = plt.scatter(x=food_sel_scaled_test.SCORE, y=scores_pred1.predictions, 
                 alpha=0.8)
plt.grid()
plt.title('OS predicted and actual results (Normalized)')
plt.xlabel('Actual Score', fontsize=16)
plt.ylabel('Predicted Score', fontsize=16)

BCsubset = (food[food.SCORE > 13])

latlon1 = pd.DataFrame(list(BCsubset.latlon.str.split()))
latlon1.columns = ['latitude', 'longitude']
latlon1['latitude'] = latlon1['latitude'].map(lambda x: str(x)[:-1])
latlon1 = latlon1.astype(float)
latlon1 = latlon1[['longitude', 'latitude']]

#Plotting geographical distribution of the points
plt.figure(figsize=(10,8))
boros.plot(alpha=0)
plt.scatter(latlon1.longitude, latlon1.latitude, s=20, alpha=0.3)
plt.title("Distribution of food stablishments @ NYC (B and C rated)", 
          fontsize = 20)
plt.ylim([40.45, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

ax3 = BCsubset['SCORE'].plot(kind='hist', bins = 20, figsize=(6,4), grid=True, 
                title="Histogram of DOH Scores (B and C)", alpha=0.8)
ax = BCsubset.plot(x='rating', y='SCORE', kind='scatter', figsize=(8,6), grid=True,
          title="DOH Score (B & C) Vs 4Square ratings", alpha=0.5) 
ax.set(xlabel="Rating", ylabel="DOH Score")          
print(spst.pearsonr(BCsubset.rating, BCsubset.SCORE))
ax1 = BCsubset.plot(x='price.tier', y='SCORE', kind='scatter', figsize=(8,6), 
                grid=True, title="DOH Score (B & C) Vs 4Square price range", alpha=0.5)
ax1.set(xlabel="Price Tier", ylabel="DOH Score")

ax2 = food['rating'].plot(kind='hist', bins = 20, figsize=(6,4), grid=True,
                 title="Histogram of 4Square ratings (B & C restaurants)", alpha=0.8)

#Subsetting and whitening the data for clustering
BC_food_sel = BCsubset[['rating','SCORE','price.tier', 'stats.checkinsCount']]
BC_food_sel['price.tier'] = BCsubset['price.tier'].fillna(value=1)
BC_food_sel1 = BCsubset[['rating', 'SCORE']]
BC_food_sel1_scaled = preprocessing.scale(BC_food_sel1)
BC_food_sel_scaled = preprocessing.scale(BC_food_sel)

#Clustering with kmeans and only rating and DOH Score
k = 2
est = cluster.KMeans(n_clusters = k, n_init = 100)
est.fit(BC_food_sel1_scaled)
colors = est.labels_.astype(np.float) 
cmap = ListedColormap(['blue','green','yellow','red', 'cyan'])
plt.figure(figsize=(10, 8))
boros.plot(alpha=0)
plt.scatter(latlon1.longitude, latlon1.latitude, s=20, c=colors, cmap=cmap, 
            alpha = 0.5)
plt.title("Clustering with K-means and 2 clusters", fontsize = 20)
plt.xlabel('Latitude', fontsize=16)
plt.ylabel('Longitude', fontsize=16)
plt.ylim([40.5, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

#Clustering with kmeans and all featrures (rating, DOH Score, price tier and
#4squared checkins)
k = 2
est = cluster.KMeans(n_clusters = k, n_init = 100)
est.fit(BC_food_sel_scaled)
colors = est.labels_.astype(np.float)
cmap = ListedColormap(['blue','green','yellow','red', 'cyan'])
plt.figure(figsize=(10, 8))
boros.plot(alpha=0)
plt.scatter(latlon1.longitude, latlon1.latitude, s=20, c=colors, cmap=cmap, 
            alpha =0.5)
plt.title("Clustering with K-means and 2 clusters", fontsize = 20)
plt.xlabel('Latitude', fontsize=16)
plt.ylabel('Longitude', fontsize=16)
plt.ylim([40.5, 40.95])
plt.xlim([-74.3, -73.6])
plt.grid()

BC_food_sel_scaled_df = pd.DataFrame(preprocessing.scale(BC_food_sel), 
                                     columns =['rating','SCORE','price.tier','stats.checkinsCount'])


#Dividing the sample into train and test sets
np.random.seed(2015)
ind=spst.bernoulli.rvs(p = 0.75, size = len(BC_food_sel))
BC_food_sel_scaled_train = BC_food_sel_scaled_df[ind==1]
BC_food_sel_scaled_test = BC_food_sel_scaled_df[ind==0] 

#Creating parameters for training RNN
n_hidden = 5 # M
n_in = 3      # D
n_out = 1     # K
n_steps = 1  # the length of each sequence
n_seq = len(BC_food_sel_scaled_train)   # the number of datapoints (i.e. sequences)

#Creating input and output arrays for training RNN         
BC_rating = np.array(BC_food_sel_scaled_train[['rating','price.tier','stats.checkinsCount']]).reshape(n_seq,n_steps,n_in)
BC_score = np.array(BC_food_sel_scaled_train['SCORE']).reshape(n_seq,n_steps,n_out)

#Creating the model and feeding it with training data set
model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=200, activation='tanh')
model.fit(BC_rating, BC_score, validation_frequency=5000)

BCguess = model.predict(BC_rating.reshape(len(BC_rating), n_in))
BCscores_pred = pd.DataFrame(BCguess)

BCscores_pred.columns = ['predictions']

BCscores_pred.predictions.plot(kind='hist', bins=20, figsize=(6,4), grid=True, 
                               title = "Histogram of predicted DOH normalized Scores B&C (IS)", alpha=0.8)

BC_food_sel_scaled_train.SCORE.plot(kind='hist', bins=20, figsize=(6,4), grid=True, 
                                    title = "Histogram of actual DOH normalized Scores B&C (IS)", alpha=0.8)


#Creating parameters for training RNN
n_hidden = 5 # M
n_in = 3      # D
n_out = 1     # K
n_steps = 1  # the length of each sequence
n_seq = len(BC_food_sel_scaled_test)   # the number of datapoints (i.e. sequences)

#Creating input and output arrays for training RNN         
BC_rating1 = np.array(BC_food_sel_scaled_test[['rating','price.tier','stats.checkinsCount']]).reshape(n_seq,n_steps,n_in)
BC_score1 = np.array(BC_food_sel_scaled_test['SCORE']).reshape(n_seq,n_steps,n_out)

BCguess1 = model.predict(BC_rating1.reshape(len(BC_rating1), n_in))
BCscores_pred1 = pd.DataFrame(BCguess1)

BCscores_pred1.columns = ['predictions']

BCscores_pred1.predictions.plot(kind='hist', bins=20, figsize=(6,4), grid=True, 
                                title = "Histogram of predicted DOH normalized Scores B&C (OS)", alpha=0.8)

BC_food_sel_scaled_test.SCORE.plot(kind='hist', bins=20, figsize=(6,4), grid=True, 
                                   title = "Histogram of actual DOH normalized Scores B&C (OS)", alpha=0.8)

fig = plt.figure(figsize=(8,6))
ax = plt.scatter(x=BC_food_sel_scaled_test.SCORE, y=BCscores_pred1.predictions, 
                 alpha=0.8)
plt.grid()
plt.title('OS predicted and actual results (Normalized)')
plt.xlabel('Actual Score', fontsize=16)
plt.ylabel('Predicted Score', fontsize=16)


