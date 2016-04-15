#!/usr/bin/env python
import requests
import pandas as pd
import os
import pickle
import time


'''
This script uses foursquare EXPLORE api to fetch the foursquare
ratings for the restaurants in the inspection data.
'''


# read csv file and drop the irrelevant column
# Merge_target.csv is essentially a table containing the restaurant name,
# inspection grade, not-so-accurate geolocation of the restaurant. I use
# these columns to query the foursquare info for each and every restaurant.
final = pd.read_csv('Merge_target.csv')
final.drop('Unnamed: 0', axis=1, inplace=True)
# scrape data from foursquare and save it in a pickle file

# create a dictionary with the restaurant index as keys and
# the query parameters as values
params_all = {}
for i in range(len(final)):
    '''
    If you plan to replicate this foursquare api query, please set up
    environment variables 'FOUR_SQUARE_CLIENT_ID' for your foursquare client
    id and 'FOUR_SQUARE_ClIENT_SECRET' for your foursquare client secret.
    If you can also reprogram this part to pass these two parts as system
    arguments. Your choice.
    '''
    params_all[i] = {'ll': final.loc[i, 'latlon'],
                     'query': final.loc[i, 'DBA'],
                     'limit': 2,
                     'client_id': os.getenv('FOUR_SQUARE_CLIENT_ID'),
                     'client_secret': os.getenv('FOUR_SQUARE_CLIENT_SECRET'),
                     'v': 20151231}
# create a dictionary restau to save the query result
restau = {}
# carry out the scraping task (api query limitation is 5000 per hour)
url = 'https://api.foursquare.com/v2/venues/explore'
n = 0
# I didn't really take care of request exception handling since this was
# small project. Feel free to build on top of this piece of code.
print 'starting the task'
for i in range(len(final)):
    restau[i] = requests.get(url=url, params=params_all[i]).json()
    n += 1
    if (n + 3000 <= len(final)) and (n % 3000.0 == 0):
        print 'parsed {0} records'.format(n)
        with open('ratings_all.pkl', 'w') as fh:
            pickle.dump(restau, fh)
            print 'the work has been saved'
        print 'resting'
        time.sleep(3600)
        print 'resume'
    elif (n + 1 == len(final)):
        print 'parsed {0} records'.format(n)
        with open('ratings_all.pkl', 'w') as fh:
            pickle.dump(restau, fh)
            print 'the work has been saved'
print 'task finished.'
