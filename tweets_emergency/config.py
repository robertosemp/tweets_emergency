#Set-up configuration information

import os
data_dir = "/home/ubuntu/tweets_emergency/data"
train_src = data_dir + "/train.csv"
test_src = data_dir + "/test.csv"

params = {'epochs' : 20,
         'batch' : 20,
         'lr' : 0.0005,
         'split' : 0.2,
          'threshold' : 0.47,
          'lemmatize' : True,
          'remove_stop' : False,
          'remove_nonalpha' : False,
          'model' : 'logReg',
         }