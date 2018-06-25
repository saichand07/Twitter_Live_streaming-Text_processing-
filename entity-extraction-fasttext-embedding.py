# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:44:40 2018

@author: Sai Chand
"""
from pathlib import Path
from gensim import fastText
#import fastText
import sklearn
import sklearn.metrics
import numpy as np
import re

root_dir = Path("..")
data_dir = root_dir / "data" / "3-entity-extraction"
notebook_dir = root_dir / "notebooks"
model_dir = data_dir / "model" 

if not model_dir.exists():
    model_dir.mkdir()
    
    
# corpus
data_path = data_dir / "twitter_las_vegas_shooting"
# Training corpus filename
input_filename = str(data_path)
# Model filename
model_filename = str(model_dir / "twitter.bin")

# Preprocessing Config
preprocess_config = {
    "hashtag": True,
    "mentioned": True,
    "punctuation": True,
    "url": True,
}

# Pattern
hashtag_pattern = "#\w+"
mentioned_pattern = "@\w+"
url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

trans_str = "!\"$%&\'()*+,-./:;<=>?[\\]^_`{|}~" + "…"
translate_table = str.maketrans(trans_str, " " * len(trans_str))

def preprocess(s):
    s = s.lower()
    if preprocess_config["hashtag"]:
        s = re.sub(hashtag_pattern, "", s)
    if preprocess_config["mentioned"]:
        s = re.sub(mentioned_pattern, "", s)
    if preprocess_config["url"]:
        s = re.sub(url_pattern, "", s)
    if preprocess_config["punctuation"]:
        s = " ".join(s.translate(translate_table).split())
    return s

# example of preprocessing
example_tweet = "RT @TheLeadCNN: Remembering Keri Lynn Galvan, from Thousand Oaks, California. #LasVegasLost https://t.co/QuvXa6WvlE https://t.co/hDF2d3Owgn"

print("Original Tweet:")
print(example_tweet)
print()
print("Preprocessed Tweet:")
print(preprocess(example_tweet))

# Preprocessing
preprocessed_data_path = data_dir / "twitter_las_vegas_shooting.preprocessed"

with data_path.open() as f:
    lines = [l.strip() for l in f.readlines()]

with preprocessed_data_path.open("w") as f:
    for l in lines:
        f.write(preprocess(l))
        f.write("\n")

# use preprocessed data as input
input_filename = str(preprocessed_data_path)
































































































