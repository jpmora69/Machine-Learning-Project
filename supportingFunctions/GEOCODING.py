#!/usr/bin/env python
import pandas as pd
import geopy


def geocoding(fname, coder=geopy.geocoders.Nominatim(), address='Address',
              n_init=0, n_end=5):
    '''
    This function takes 5 arguments: fname, coder, col, n_init, n_end
    fname: the pandas dataframe you want to geocode
    coder: geocoder of your choice: e.g. Nominatim, GoogleV3, GeoNames
    address: the column containing the address you want to geocode
    n_init: the starting index
    n_end: the finishing index
    '''
    # prepare the table: create two new columns: 'lat' and 'lon'
    fname['lat'] = None
    fname['lon'] = None
    fname['lat'] = fname.loc[n_init: n_end,
                             address].apply(lambda x:
                                            coder.geocode(x).latitude)
    fname['lon'] = fname.loc[n_init: n_end,
                             address].apply(lambda x:
                                            coder.geocode(x).longitude)
