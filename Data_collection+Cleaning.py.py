from __future__ import absolute_import, print_function

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
#from ipywidgets import IntProgress
#import requests
import folium
#import os
import tweepy
#import uuid
import json
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point
from IPython.core.display import display



# Load sklearn and LDA (Latent Dirichlet Allocation):
from sklearn.decomposition import LatentDirichletAllocation

# Load word tokenisers and vectorisers:
from sklearn.feature_extraction.text import CountVectorizer



# == OAuth Authentication ==
#
# This mode of authentication is the new preferred way
# of authenticating with Twitter.

# The consumer keys can be found on your application's Details
# page located at https://dev.twitter.com/apps (under "OAuth settings")
consumer_key="Enter_key"
consumer_secret="Enter_secret"

# The access tokens can be found on your applications's Details page located at https://dev.twitter.com/apps (located under "Your access token")
 

access_token="enter_token"
access_token_secret="enter_token_secret"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# If the authentication was successful,
# you can see the name of the account print out
print(api.me().name)

api = tweepy.API(auth)
# you can enter any keywords (required on topic or intrest) 
keywords = ['earthquake', 'quake', 'magnitude', 'epicenter', 'magnitude', 'aftershock']

# Collect 100 tweets using the keywords:
search_results = api.search(q=' OR '.join(keywords), count=100)

df = pd.DataFrame([ {'id': result.id, 'created_at': result.created_at, 'user': '@'+result.user.name, 'text': result.text } for result in  search_results])[["id", "created_at", "user", "text"]]

display(df.head())
    
# If the application settings are set for "Read and Write" then
# this line should tweet out the message to your account's
# timeline. The "Read and Write" setting is on https://dev.twitter.com/apps

#api.update_status(status='Updating using OAuth authentication via Tweepy!')

if search_results is not []:
    # Print the first Tweet as JSON:
    print(json.dumps(search_results[0]._json, indent=2))

df = pd.DataFrame([ {'id': result.id, 'created_at': result.created_at, 'user': '@'+result.user.name, 'text': result.text } for result in  search_results])[['id', 'created_at', 'user', 'text']]
display(df.head())

# What is the weather 500mm around Lyon?: 
keywords2 = ['weather' , 'forcast', 'sun', 'rain', 'clouds', 'storm']

# Only in english please !
lang = 'en'

# Get tweets around Lyon (latitide,longitude,radius):
geocode = '45.76,4.84,500km'

# Collect 1500 tweets using the keywords:
search_results2 = api.search(q=' OR '.join(keywords2), geocode=geocode, lang=lang, count=1500)


# Convert to GeoPandas:
df2 = pd.DataFrame([ {'id': result.id, 'created_at': result.created_at, 'user': '@'+result.user.name, 'text': result.text, 'geometry': result.coordinates } for result in  search_results2])[['id', 'created_at', 'user', 'text', 'geometry']]
df2['geometry'] = df2['geometry'].apply(lambda coords: np.nan if coords is None else Point(coords['coordinates']))
df2 = df2.dropna() # Remove documents without geometry point (the twitter API may obtain location using user details rather than the tweet location.).
df2 = GeoDataFrame(df2, crs = {'init': 'epsg:2263'})

display(df2.head())

# Create a map:
results2_map = folium.Map([45.76, 4.84],
                  zoom_start=6,
                  tiles='Stamen Toner')

# Iterate over documents and add them the amp:
for index, document in tqdm_notebook(df2.iterrows()):
    
    # Perform very simple matching of words for displaying the weather:
    if 'sun' in document['text'].lower() or 'clear' in document['text'].lower():
        icon_img = 'sun-o'
        color = 'orange'
        
    elif 'rain' in document['text'].lower() or 'showers' in document['text'].lower():
        icon_img = 'umbrella'
        color = 'blue'
        
    elif 'strom' in document['text'].lower():
        icon_img = 'bolt'
        color = 'black'
    
    elif 'cloud' in document['text'].lower():
        icon_img = 'cloud'
        color = 'gray'
        
    else:
        icon_img = 'info-circle'
        color = 'green'
    
    icon = folium.Icon(color=color, icon_color='white', icon=icon_img, angle=0, prefix='fa')
    folium.Marker([document['geometry'].y, document['geometry'].x], popup=document['text'], icon=icon).add_to(results2_map)

display(results2_map)

# Create listener:
class TwitterStreamListener(tweepy.StreamListener):
    
    def __init__(self, api=None, max_count=1000):
        super(TwitterStreamListener, self).__init__(api)
        
        self.max_count = max_count
        self.count = 0
        self._data = []
        
        if self.max_count is not None:
            self._pbar = tqdm_notebook(total=self.max_count, desc='Collecting tweets.')
            self._pbar.clear()
      
    # Retrun the data as a dataframe:
    @property
    def data(self):
        results = self._data
        df = pd.DataFrame([ {'id': result.id, 'created_at': result.created_at, 'user': '@'+result.user.name, 'text': result.text, 'geometry': result.coordinates } for result in  results])[['id', 'created_at', 'user', 'text', 'geometry']]
        df['geometry'] = df['geometry'].apply(lambda coords: np.nan if coords is None else Point(coords['coordinates']))
        df = GeoDataFrame(df, crs = {'init': 'epsg:2263'})
        
        return df
        
        
    # Do something when a tweet is received:
    def on_status(self, status):
        
        # Stop collecting when max_count is reached:
        self.count = self.count + 1
        #print(str(self.count) + ': ' + status.text)
        self._data.append(status)
        
        if self.max_count is not None:
            self._pbar.update()

            if self.count == self.max_count:
                self._pbar.close()
                return False

# Create a stream listener that collect 10 tweet:
stream_listener = TwitterStreamListener(max_count=10)
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

# Get earthquake related information:
keywords3 = ['earthquake', 'quake', 'magnitude', 'epicenter', 'magnitude', 'aftershock']

# Go fetch:
stream.filter(track=keywords3)
display(stream_listener.data.head())

# Access the datasets from your Data repository 
files = {
 'Philipinnes floods': '../data//2012_Philipinnes_floods-tweets_labeled.csv',    
 'Alberta floods': '../data//2013_Alberta_floods-tweets_labeled.csv',
 'Colorado floods': '../data//2013_Colorado_floods-tweets_labeled.csv',
 'Manila floods': '../data//2013_Manila_floods-tweets_labeled.csv',  
 'Queensland floods': '../data//2013_Queensland_floods-tweets_labeled.csv',
 'Sardinia floods': '../data//2013_Sardinia_floods-tweets_labeled.csv' }

frames = []
for event, f in tqdm_notebook(files.items(), desc='Fetch data'):
    frames.append(pd.read_csv(f))
    
data = pd.concat(frames, keys=files.keys())
display(data.head())

# We split each documents by words using the default Sklearn tokenizer:
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
vects = count_vectorizer.fit_transform(data[' Tweet Text'])
vocabulary = count_vectorizer.get_feature_names()

# Number of tokens:
print('Vocabulary size: '+str(len(vocabulary)))

# We want to extract 5 topics:
nb_topics = 5
lda = LatentDirichletAllocation(n_components=nb_topics, learning_method='batch', random_state=42)


# We fit the model to the textual data: 
lda.fit(vects)

n_top_words = 30
topic_words = {}

# Get the top words for each topic:
for topic, comp in enumerate(lda.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words['Topic #'+str(topic)] = [vocabulary[i] for i in word_idx]
    
topic_words = pd.DataFrame.from_dict(topic_words)
display(topic_words)

topics = ['thoughts', 'needs_and_volunteering', 'victims', 'donnations', 'floods']
topic_words.columns = topics
display(topic_words.head())

queensland_data = pd.read_csv('../data/2-data-collection/2013_Queensland_floods-tweets_labeled.csv')
topic_preds = [topics[p] for p in np.argmax(lda.transform(count_vectorizer.transform(queensland_data[' Tweet Text'])), axis=1)]

pd.Series(topic_preds).value_counts(normalize=True).plot.barh()
plt.title('Topic distribution in Queensland floods', fontsize=20)
plt.xlabel('Proportion')
plt.ylabel('Topic')
plt.show()

classified_queensland_data = queensland_data.assign(topic=topic_preds)
classified_queensland_data.loc[classified_queensland_data['topic'] == 'victims']












