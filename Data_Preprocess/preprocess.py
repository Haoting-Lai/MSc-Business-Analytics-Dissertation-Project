#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langid')
get_ipython().system('pip install langdetect')
get_ipython().system('pip install nltk')
get_ipython().system('pip install unidecode')
get_ipython().system('pip install contractions')
get_ipython().system('pip install spacy')
get_ipython().system('pip install wordsegment #https://github.com/grantjenks/python-wordsegment')


# In[2]:


import pandas as pd
import numpy as np
import json
import langid
import gensim
import re
import os
import joblib
import spacy
import string
import unidecode
import contractions
import requests
import io
import ast
import wordsegment
import nltk

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from langdetect import detect
from wordsegment import load, segment

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import ngrams
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[3]:


lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_description(text):
    text = re.sub(r'^\[|\]$', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

def remove_accented_chars(text):
    if text is not None:
        text = unidecode.unidecode(text)
    return text

def expand_contractions(text):
    if text is not None:
        text = contractions.fix(text)
    return text

def segment_text(text, max_len=20):
    words = []
    for part in text.split():
        if len(part) > max_len:
            words.extend(segment(part[i:i+max_len]) for i in range(0, len(part), max_len))
        else:
            words.extend(segment(part))
    return words

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_text(text):
    lemma_words = []
    for word in text:
        word_str = "".join(word)
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(word_str))
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemma_words.append(''.join(lemmatized_sentence))
    return lemma_words


# In[7]:


def preprocess(df):
    df_norm = pd.json_normalize(df['_source'])
    df = pd.concat([df.drop('_source', axis=1), df_norm], axis=1)
    df = df.drop(['_index', '_type', '_score', '_ignored', 'fields','bip_data.ram'], axis=1)
    raw_data = df.rename(columns={
        'openaire.maintitle': 'maintitle',
        'openaire.description': 'description',
        'bip_data.citationcount': 'citationcount',
        'bip_data.threeyrcitationcount': 'threeyrcitationcount',
        'bip_data.attrank': 'attrank',
        'bip_data.pagerank': 'pagerank'
    })
    raw_data['description'] = raw_data['description'].apply(lambda x: clean_description(x[0]) if isinstance(x, list) and len(x) > 0 else np.nan)
    raw_data = raw_data.dropna(subset=['description'])
    raw_data['description'] = raw_data['description'].str.lower()
    tags_to_replace = ['jatsbold','jatsp', 'jatstitle', 'jatssec', 'jatspthe', 'jatsitalic', 'executive overview', 'abstract', 'synopsis', 'background', 'aims','introduction', 'aim']
    raw_data['description'] = raw_data['description'].replace(tags_to_replace, '', regex=True)
    raw_data['description'] = raw_data['description'].str.replace(r'\s+', ' ', regex=True).str.strip()
    filtered_df = raw_data
    filtered_df['clean_description'] = filtered_df['description'].apply(remove_stopwords)
    filtered_df['description'] = filtered_df['description'].apply(lambda x: remove_accented_chars(x) if pd.notnull(x) else np.nan)
    filtered_df['description'] = filtered_df['description'].apply(lambda x: expand_contractions(x) if pd.notnull(x) else np.nan)
    load()
    filtered_df['clean_description'] = filtered_df['clean_description'].fillna('')
    filtered_df['clean_description'] = filtered_df['clean_description'].astype(str)
    filtered_df['clean_description'] = filtered_df['clean_description'].apply(segment_text)
    filtered_df['lemmatized_description'] = filtered_df['clean_description'].apply(lemmatize_text)
    return filtered_df

