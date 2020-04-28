import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

def load_CSV(src):
    
    #load a CSV file from the source address. returns a pandas dataframe
    logger = logging.getLogger("logger") 
    
    try:
        data = pd.read_csv(src, sep = ',')
    except Exception as e:
        logger.exception("Unable to find  CSV. Error: %s", e)
    return data


def create_freq_table(pd_df):
    
    # create a dictionary of i elements where is the # of columns in the pandas dataframe. 
    # each column i has an associated dictionary of unique elements and their frequency count
    # looping makes sense since we need to go hrough every element in the pd dataframe
    # alternatively we could create masks of the set of elements that exist (could be faster)
    # could also do map-reduce but we dont have a cluster
    
    columns = pd_df.columns
    columns_dict = {}
    for i in columns:
        column_list = list(pd_df[i])
        columns_dict[i] = create_column_freq(column_list)
    return columns_dict



def create_column_freq(column_list):
    
    # column_list: a list of data. In this case - a list version of one column of a pandas dataframe 
    # output: a dictionary with each unique element in the column_list and its associated count. i.e.
    #         {'a': 10, 'b': 1, ... }
    
    column_dict = {}
    for j in column_list:
        if j in column_dict:
            column_dict[j] = column_dict[j] + 1
        else:
            column_dict[j] = 1
    return column_dict


def create_unique_sets(pd_df):
    
    # pd_df: pandas dataframe - train data
    # output: a dictionary of i sets (elements) where i is each column in the pandas dataframe pd_df
    
    columns = (pd_df.columns)
    columns_dict_set = {}
    for i in columns:
        columns_dict_set[i] = set(pd_df[i])
    return columns_dict_set


def unique_set_perc(unique_sets, size):
    
    # unique_sets: a pandas dataframe composed of i sets of unique elements
    # size: size of the original number of rows in the dataframe
    # output: a dictionary with the proportion of unique elements per column
    
    dict_unique_perc = {}
    for key in unique_sets.keys():
        dict_unique_perc[key] = len(unique_sets[key]) / size
    return dict_unique_perc


def graph_frequencies(freq_table, perc_unique, limit, size):
    
    # freq_table: a nested dictionary with i column-elements each of which is a 
    # dictionary with the element-name and the element-count
    # perc_unique: simpe dictionary with column-elements and their % of unique elements. 
    # used to control sparse data
    # limit: threshold to decide which columns to graph. If data is too unique dont graph
    # output: n graphs that satisfy the threshold above
    
    for key in freq_table.keys():     
        if perc_unique[key] < limit:
            size_categories = round(10 * np.log(perc_unique[key] * size))
            if size_categories >= 8:
                font = round(size_categories / 4)
            else:
                font = round(size_categories * 2)
            df = pd.DataFrame({'term' : list(freq_table[key].keys()), 'freq' : list(freq_table[key].values())})
            df = df.sort_values('freq')
            df.plot(kind='barh', legend=False, align = 'center', figsize=(8, size_categories), fontsize = font)
            plt.yticks(range(len(freq_table[key].keys())), list(df['term']))

          


def create_masterset(lemmatized_texts):
    masterset = set()
    for row in lemmatized_texts:
        for word in row:
            masterset.add(word)
    return masterset


def create_mastercount(lemmatized_texts, target):
    
    logger = logging.getLogger("logger") 
    
    try:
        len(target) == len(lemmatized_texts)
    except Exception as e:
        logger.exception("mismatch in size. Error: %s", e)
        
    dict_by_target = {}
    dict_by_target[0] = {}
    dict_by_target[1] = {}
    count = 0
    for row in lemmatized_texts:
        for word in row:
            if word in dict_by_target[target[count]]:
                dict_by_target[target[count]][word] = dict_by_target[target[count]][word] + 1
            else:
                dict_by_target[target[count]][word]  = 1
        count+=1
        
    pos_wordcount = pd.DataFrame({'term' : list(dict_by_target[1].keys()), 'freq' : list(dict_by_target[1].values())}).sort_values('freq', ascending = False)
    neg_wordcount = pd.DataFrame({'term' : list(dict_by_target[0].keys()), 'freq' : list(dict_by_target[0].values())}).sort_values('freq', ascending = False)
    
    return pos_wordcount, neg_wordcount


def graph_word_count(pd_df, number):
    top = pd_df[0:number]
    top.plot(kind='bar', legend=False, align = 'center', figsize=(18, 8), fontsize = 14)
    plt.xticks(range(number), list(top['term']))
    
    
def create_len_histogram(lemmatized_texts, target):
        
    logger = logging.getLogger("logger") 
        
    try:
        len(target) == len(lemmatized_texts)
    except Exception as e:
        logger.exception("mismatch in size. Error: %s", e)
        
    dict_by_target = {}
    dict_by_target[0] = []
    dict_by_target[1] = []
    count = 0
    for row in lemmatized_texts:
        dict_by_target[target[count]].append(len(row))
        count+=1
    return dict_by_target[1], dict_by_target[0]

