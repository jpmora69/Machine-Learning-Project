[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Elixeus/Machine-Learning-Project/blob/master/LICENSE.md)
# About our project
Our main goal of this project is to apply the machine learning techniques to data available in the urban context. For this project, we are particularly interested in the restaurant hygene condition in New York City. The Big Apple has more than [20,000 restaurants by July of 2015](http://http://www.wsj.com/articles/new-york-city-restaurants-multiply-despite-high-profile-closures-1412816142). But not every restaurant is ready to provide the clients with a service both safe and delicious. According to the Department of Health and Mental Hygene (DOHMH), only 72% of the restaurants among all the restaurants inspected had an A grade. So the food safety remains a concern for New Yorkers who choose to eat outside.

On the other hand, researchers has applied machine learning techniques to data available from other sources, especially mass-based restaurant recommendation software such as Yelp or Foursquare. See [Jin and Leslie 2003](http://qje.oxfordjournals.org/content/118/2/409.short), [Jin and Leslie 2009](https://www.aeaweb.org/articles?id=10.1257/mic.1.1.237), [Kang et al. 2013](http://www3.cs.stonybrook.edu/~junkang/hygiene/). These previous research provided a new approach: combine the restaurant ratings and other features from the restaurant recommendation software data, and find correlation between these features and restaurant inspection grades. For our project, we use data provided by foursquare and try to establish correlation between ratings and inspection results. Also we would like to visualize the risky areas in New York and try to find spatial patterns. We would also like to run a spatial autocorrelation analysis to see if restaurants nearby affect each other's food hygene performance.
# Why this repo exists
This is the repo for our Machine Learning final group project at [CUSP, NYU](http://cusp.nyu.edu/). It serves as a platform of code version control, code exchange, code safehouse and reproducibility.
# License:
This project is licensed under the [MIT License](https://github.com/Elixeus/Machine-Learning-Project/blob/master/LICENSE.md).
# TODO:
- [x] Scrape data from Foursquare
- [ ] Select 2015 Inspection Data
- [x] Geocode Restaurant Addresses
- [x] Merge rating data with DOHMH inspection data
- [ ] Neural network classification
- [ ] Spatial autocorrelation analysis
