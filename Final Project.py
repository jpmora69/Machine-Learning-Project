# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:38:04 2016

@author: juanpablo
"""

import pandas as pd
import geopandas as gpd
import shapely.geometry as geom
import mplleaflet
import pylab as pl
import numpy as np

foursquare = pd.read_csv('NY_all.csv')
food = foursquare[(foursquare.genCategory == "Food") & (foursquare.rating !=-1)]
food[['lon']] = food[['lon']].astype(float)
longlat = food[['lon', 'lat']]
points = map(geom.Point, longlat.as_matrix().tolist())
gpos = gpd.GeoSeries(points)
epsg4326 = {'init': 'epsg:4326'}
gpos.crs = epsg4326
projectedPos = gpos.to_crs(epsg=2263)
shapefile = gpd.read_file('nybb_16a/nybb1.shp')
intersect = projectedPos.intersects(shapefile.unary_union)# define the intersection of
#all polygon with each geopolypoints.
np.sum(intersect) # find out the number of true values in the array
intersectdf = pd.DataFrame(intersect, columns = ['boolean'])
food = food.reset_index()
del food['index']
food = pd.concat([food, intersectdf], axis=1)
food.boolean = food.boolean.astype(str)
food = food[(food.boolean == "True")]
food[['lon']] = food[['lon']].astype(float)
longlat = food[['lon', 'lat']]
points = map(geom.Point, longlat.as_matrix().tolist())
gpos = gpd.GeoSeries(points)
epsg4326 = {'init': 'epsg:4326'}
gpos.crs = epsg4326
projectedPos = gpos.to_crs(epsg=2263)
#projectedPos.buffer(1000, resolution=1).plot(figsize=(8,8))
#mplleaflet.display(crs=projectedPos.crs)
pl.figure()
projectedPos.plot(k=100, alpha=1.0, figsize=(8,8))
pl.grid()
pl.show()