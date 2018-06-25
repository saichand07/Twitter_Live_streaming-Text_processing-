from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import pandas as pd
import tweepy
from geopandas import GeoDataFrame
from shapely.geometry import Point
import numpy as np
from IPython.core.display import display

from time import time, sleep

# Go to http://apps.twitter.com and create an application
# The consumer key and secret will be generated for you after
consumer_key="enter the consumer key "
consumer_secret="password please"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token="enter your access_token_number"
access_token_secret="enter your access_token_secret"

class StdOutListener(StreamListener):
    """ A listener handles the tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, count = 10):
        print(count)
        return True

    def on_error(self, status):
        print(status)
        
sleep(5)
        
if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=['earthquake', 'quake', 'magnitude', 'epicenter', 'magnitude', 'aftershock'])

time.sleep(.300)

api = tweepy.API(auth)
keywords = ['earthquake', 'quake', 'magnitude', 'epicenter', 'magnitude', 'aftershock']

# Collect 100 tweets using the keywords:
search_results = api.search(q=' OR '.join(keywords), count=10)

df = pd.DataFrame([ {'id': result.id, 'created_at': result.created_at, 'user': '@'+result.user.name, 'text': result.text } for result in  search_results])[['id', 'created_at', 'user', 'text']]
df.display(df.head())

# What is the weather 500mm around Lyon?: 
keywords2 = ['weather' , 'forcast', 'sun', 'rain', 'clouds', 'storm']

# Only in english please !
lang = 'en'

# Get tweets around Lyon (latitide,longitude,radius):
geocode = '45.76,4.84,500km'

# Collect tweets using the keywords:
search_results2 = api.search(q=' OR '.join(keywords2), geocode=geocode, lang=lang, count=1500)


# Convert to GeoPandas:
df2 = pd.DataFrame([ {'id': result.id, 'created_at': result.created_at, 'user': '@'+result.user.name, 'text': result.text, 'geometry': result.coordinates } for result in  search_results2])[['id', 'created_at', 'user', 'text', 'geometry']]
df2['geometry'] = df2['geometry'].apply(lambda coords: np.nan if coords is None else Point(coords['coordinates']))
df2 = df2.dropna() # Remove documents without geometry point (the twitter API may obtain location using user details rather than the tweet location.).
df2 = GeoDataFrame(df2, crs = {'init': 'epsg:2263'})

display(df2.head())
    