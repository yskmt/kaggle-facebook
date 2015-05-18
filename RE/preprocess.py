# """
# preprocess.py

# Facebook Recruiting IV: Human or Robot?

# author: Yusuke Sakamoto

# """

import numpy as np
import pandas as pd


###########################################################################
## CLEANUP
###########################################################################

# trainfile = "data/train.csv"
# testfile = "data/test.csv"
# bidsfile = "data/bids.csv"

# print "Reading raw csv files..."
# traindata = pd.read_csv(trainfile)
# bidsdata = pd.read_csv(bidsfile)
# testdata = pd.read_csv(testfile)

# print "Separating traindata into bots and human data..."
# botsdata = traindata[traindata.outcome == 1]
# humansdata = traindata[traindata.outcome == 0]

# num_bots = botsdata.shape[0]

# print "Collecting the bids by bots..."
# bids_bots = []
# for i in range(num_bots):
#     bids_bots.append(bidsdata[bidsdata.bidder_id == botsdata.iloc[i, 0]])
# bids_bots = pd.concat(bids_bots)

# print "Collecting the bids by humans..."
# bids_humans = []
# num_humans = humansdata.shape[0]
# for j in range(num_humans):
#     if (j % 50) == 0:
#         print "%d/%d" % (j, num_humans)
#     bids_humans.append(bidsdata[bidsdata.bidder_id == humansdata.iloc[j, 0]])
# bids_humans = pd.concat(bids_humans)

# print "Collecting relevant bids by bidders in test data",\
#     " (not sure human or bot)..."
# bids_test = []
# num_test = testdata.shape[0]
# for k in range(num_test):
#     if (k % 50) == 0:
#         print "%d/%d" % (k, num_test)
#     bids_test.append(bidsdata[bidsdata.bidder_id == testdata.iloc[k, 0]])
# bids_test = pd.concat(bids_test)

# print "Dropping unncessary columns..."

# # drop bid_id as it is duplicate
# bids_bots.drop('bid_id', axis=1, inplace=True)
# bids_bots.index.name = 'bid_id'

# bids_humans.drop('bid_id', axis=1, inplace=True)
# bids_humans.index.name = 'bid_id'

# bids_test.drop('bid_id', axis=1, inplace=True)
# bids_test.index.name = 'bid_id'

# print "Saving bids data..."
# bids_bots.to_csv('data/bids_bots.csv')
# bids_humans.to_csv('data/bids_humans.csv')
# bids_test.to_csv('data/bids_test.csv')

###########################################################################
# print "Read bids data"
# bids_bots = pd.read_csv('bids_bots.csv')
# bids_human = pd.read_csv('bids_human.csv')
# bids_test = pd.read_csv('bids_test.csv')


###########################################################################
## ANALYSIS
###########################################################################

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

