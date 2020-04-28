import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
            
    
def tokenize_text(text, remove_nonalpha):
    #given a sentence, converts it to a list of tokens
    if remove_nonalpha:
        return nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower())
    else:
        return nltk.tokenize.word_tokenize(text.lower())
    
    
def lemmatize_tokens(list_tokens):
    lemmatizer = WordNetLemmatizer() 
    return [lemmatizer.lemmatize(w.lower()) for w in list_tokens]

def remove_STOP(list_tokens):
    new = []
    for word in list_tokens:
        if word.lower() not in set(stopwords.words('english')):
            new.append(word.lower())
    return new

def process_textlist(list_obj, lemmatize = True, remove_stop = True, remove_nonalpha = False):
    lemmatized_texts = []
    for sentence in list_obj:
        new_list = tokenize_text(sentence, remove_nonalpha)
        if lemmatize:
            new_list = lemmatize_tokens(new_list)
        if remove_stop:
            new_list = remove_STOP(new_list)
        lemmatized_texts.append(new_list)
    return lemmatized_texts
    