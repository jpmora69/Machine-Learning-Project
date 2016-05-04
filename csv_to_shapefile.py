
# coding: utf-8

# In[ ]:

#http://geospatialpython.com/2015/08/csv-to-shapefile.html
import csv
import shapefile as shp
import pandas as pd


# In[ ]:

#read in data
data =pd.read_csv('food_geo3.csv')
data.head()


# In[ ]:

# Create a polygon shapefile writer
w = shp.Writer(shp.POINT)
w.autoBalance = 1
# Add our fields
#w.field('Unnamed: 0','C', '40')
w.field('DBA','C', '50')
w.field('rating','F', '40')
w.field('stats.checkinsCount','F', '40')
w.field('price.message','C', '40')
w.field('price.tier','F', '40')
w.field('CAMIS','C', '40')
w.field('SCORE','C', '40')
w.field('sub_cat','C', '40')
w.field('lat','F', '40')
w.field('lon','F', '40')
w.field('latlon','C', '40')
#w.field('geometry', 'C', '40')

# Open the csv file and set up a reader
with open("food_geo3.csv") as p:
    reader = csv.DictReader(p)
    for row in reader:
        # Add records for each point for name and area
        w.record(row['DBA'],row['rating'], row['stats.checkinsCount'],
                row['price.tier'], row['CAMIS'],row['SCORE'],
                 row['sub_cat'], row['lat'], row['lon'],row['latlon'])
        # parse the coordinate string
        wkt = row['geometry'][7:-1]
        # break the coordinate string in to x,y values
        coords = wkt.split(",")
        # set up a list to contain the coordinates
        part = []
        # convert the x,y values to floats
        for c in coords:
            y,x = c.split()
            part.append([float(x),float(y)])
        # create a polygon record with the list of coordinates.
        w.point(float(x),float(y))

# save the shapefile!
w.save("newer_polys.shp")


# In[ ]:



