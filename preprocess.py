"""
preprocess.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd


trainfile = "data/train.csv"
testfile = "data/test.csv"
bidsfile = "data/bids.csv"

traindata = pd.read_csv(trainfile)
bidsdata = pd.read_csv(bidsfile)
testdata = pd.read_csv(testfile)

botdata = traindata[traindata.outcome == 1]
humandata = traindata[traindata.outcome == 0]

num_bots = botdata.shape[0]

print "collecting the bids by bots..."
bids_bots = []
for i in range(num_bots):
    bids_bots.append(bidsdata[bidsdata.bidder_id == botdata.iloc[i, 0]])
bids_bots = pd.concat(bids_bots)

print "collecting the bids by human..."
bids_human = []
num_humans = humandata.shape[0]
for j in range(num_humans):
    bids_human.append(bidsdata[bidsdata.bidder_id == humandata.iloc[j, 0]])
bids_human = pd.concat(bids_human)

print "collecting the bids from test data (not sure human or bots)..."
bids_test = []
num_test = testdata.shape[0]
for j in range(num_test):
    bids_test.append(bidsdata[bidsdata.bidder_id == testdata.iloc[j, 0]])
bids_test = pd.concat(bids_test)

bids_bots = pd.read_csv('data/bids_bots.csv')
bids_human = pd.read_csv('data/bids_human.csv')
bids_test = pd.read_csv('data/bids_test.csv')

print "dropping unnecessary columns..."
bids_bots.drop('Unnamed: 0', axis=1, inplace=True)
bids_human.drop('Unnamed: 0', axis=1, inplace=True)
bids_test.drop('Unnamed: 0', axis=1, inplace=True)

print "saving bids data..."
bids_bots.to_csv('data/bids_bots.csv', index=False)
bids_human.to_csv('data/bids_human.csv', index=False)
bids_test.to_csv('data/bids_test.csv', index=False)

bids_bots = pd.read_csv('bids_bots.csv')
bids_human = pd.read_csv('bids_human.csv')
bids_test = pd.read_csv('bids_test.csv')

# # train classes
# outcome = np.concatenate((np.ones(num_bots), np.zeros(num_humans)))

# bids_train = pd.concat([bids_bots, bids_human])

# # drop useless-looking labels
# bidder_ids_train = bids_train['bidder_id']
# bidder_ids_test = bids_test['bidder_id']
# bids_train = bids_train.drop(['bid_id', 'bidder_id', 'ip'], axis=1)
# bids_test = bids_test.drop(['bid_id', 'bidder_id', 'ip'], axis=1)

# # same data pre-analysis

# bids_train.keys()
# #  ['auction', 'merchandise', 'device', 'time', 'country', 'url']

# num_train = bids_train.shape[0]          # 3071224
# len(bids_train['auction'].unique())      # 12740
# len(bids_train['merchandise'].unique())  # 10
# len(bids_train['device'].unique())       # 5729
# len(bids_train['time'].unique())         # 742669
# len(bids_train['country'].unique())      # 199
# len(bids_train['url'].unique())          # 663873
