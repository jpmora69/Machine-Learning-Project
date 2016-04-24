# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:42:23 2016

@author: juanpablo
"""

import pickle 
import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
#Importing the file and applying the json_normalize funtion to the info
# we want in the dictionary (placed in the route that I used)
file1 = open('ratings_1.pkl', 'rb')
file2 = open('ratings_2.pkl', 'rb')
file3 = open('ratings_3.pkl', 'rb')
data1 = pickle.load(file1)
data2 = pickle.load(file2)
data3 = pickle.load(file3)
#Iterating trough the dictionary keys to get all of the restaurants
food_data1 = []
emptydf = pd.DataFrame({'A' : [np.nan]})
for i in range (0, len(data1)):
    try:
        datadf = pd.io.json.json_normalize(data1[i]['response']['groups'][0]['items'][0]['venue'])
    except IndexError:
        datadf = emptydf
    food_data1.append(datadf)    
food_df1 = pd.concat(food_data1)
#Iterating trough the dictionary keys to get all of the restaurants
food_data2 = []
emptydf = pd.DataFrame({'A' : [np.nan]})
for i in range(len(data1), (len(data1)+len(data2))):
    try:
        datadf = pd.io.json.json_normalize(data2[i]['response']['groups'][0]['items'][0]['venue'])
    except Exception:
        pass
    except IndexError:
        datadf = emptydf
    food_data2.append(datadf)    
food_df2 = pd.concat(food_data2)
#Iterating trough the dictionary keys to get all of the restaurants
food_data3 = []
emptydf = pd.DataFrame({'A' : [np.nan]})
for i in range(12000, 18689):
    try:
        datadf = pd.io.json.json_normalize(data3[i]['response']['groups'][0]['items'][0]['venue'])
    except Exception:
        pass
    except IndexError:
        datadf = emptydf
    food_data3.append(datadf)    
food_df3 = pd.concat(food_data3)
#Subsetting the columns we want
food_df1 = food_df1[['name', 'rating', 'stats.checkinsCount', 'price.message', 'price.tier']]
food_df2 = food_df2[['name', 'rating', 'stats.checkinsCount', 'price.message', 'price.tier']]
food_df3 = food_df3[['name', 'rating', 'stats.checkinsCount', 'price.message', 'price.tier']]
#Resetting the index and deleting the previous index column
food_df1 = food_df1.reset_index()
del food_df1['index']
food_df2 = food_df2.reset_index()
del food_df2['index']
food_df3 = food_df3.reset_index()
del food_df3['index']
#Joining the three parts of the data
food_int = food_df1.append([food_df2, food_df3], ignore_index=True)
#Removing non-matching records from the 4squared dataset
food_int = food_int.drop(labels=[18688], axis=0)
#Importing the DOH dataset
health = pd.read_csv('Merge_target1.csv')
#Removing non matching records from DOH dataset
health = health.drop(labels=[18688, 18689], axis=0)
#Reading categories data set
categories = pd.read_csv('sub_cat.csv')
#Printing data set tail
categories.tail(10)
#Concatenating all dataframes
food_final = pd.concat([pd.concat([food_int, health], axis=1), categories], axis=1)
#Subsetting the data set based on existing 4squared ratings
food_final1 = food_final[food_final['rating'] > 0]
#Deleting former index column
del(food_final1['Unnamed: 0'])
#Reseting index
food_final1 = food_final1.reset_index()
#Deleting former index column
del(food_final1['index'])
#Exporting results to CSV file
food_final1.to_csv('food_final.csv', encoding='utf-8')