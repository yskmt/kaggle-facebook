"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd


# trainfile = "data/train.csv"
# testfile = "data/test.csv"
# bidsfile = "data/bids.csv"

# traindata = pd.read_csv(trainfile)
# bidsdata = pd.read_csv(bidsfile)
# testdata = pd.read_csv(testfile)

# botdata = traindata[traindata.outcome == 1]
# humandata = traindata[traindata.outcome == 0]

num_bots = botdata.shape[0]

# print "collecting the bids by bots..."
# bids_bots = []
# for i in range(num_bots):
#     bids_bots.append(bidsdata[bidsdata.bidder_id == botdata.iloc[i, 0]])
# bids_bots = pd.concat(bids_bots)

# print "collecting the bids by human..."
# bids_human = []
num_humans = humandata.shape[0]
# for j in range(num_humans):
#     bids_human.append(bidsdata[bidsdata.bidder_id == humandata.iloc[j, 0]])
# bids_human = pd.concat(bids_human)

# print "collecting the bids from test data (not sure human or bots)..."
# bids_test = []
# num_test = testdata.shape[0]
# for j in range(num_test):
#     bids_test.append(bidsdata[bidsdata.bidder_id == testdata.iloc[j, 0]])
# bids_test = pd.concat(bids_test)

# print "saving bids data..."
# bids_bots.to_csv('bids_bots.csv')
# bids_human.to_csv('bids_human.csv')
# bids_test.to_csv('bids_test.csv')


bids_bots = pd.read_csv('bids_bots.csv')
bids_human = pd.read_csv('bids_human.csv')
bids_test = pd.read_csv('bids_test.csv')

# train classes
outcome = np.concatenate((np.ones(num_bots), np.zeros(num_humans)))

bids_train = pd.concat([bids_bots, bids_human])

# drop useless-looking labels
bidder_ids_train = bids_train['bidder_id']
bidder_ids_test = bids_test['bidder_id']
bids_train = bids_train.drop(['bid_id', 'bidder_id', 'ip'], axis=1)
bids_test = bids_test.drop(['bid_id', 'bidder_id', 'ip'], axis=1)

# same data pre-analysis

bids_train.keys()
#  ['auction', 'merchandise', 'device', 'time', 'country', 'url']

num_train = bids_train.shape[0]          # 3071224
len(bids_train['auction'].unique())      # 12740
len(bids_train['merchandise'].unique())  # 10
len(bids_train['device'].unique())       # 5729
len(bids_train['time'].unique())         # 742669
len(bids_train['country'].unique())      # 199
len(bids_train['url'].unique())          # 663873

# brute-force dummy labeling will create 1,425,220 labels
# time can be categorize into smaller groups?
# cluster the dataset using some unsupervised learning technique?

# first try with smaller data sets
bids_train_small = bids_train.drop(['auction', 'device', 'time', 'url'], axis=1)
bids_test_small = bids_test.drop(['auction', 'device', 'time', 'url'], axis=1)

# create dummy labels
dummies = []
for key in bids_train_small.keys():
    dummies.append(pd.get_dummies(bids_train_small[key]))
bids_train_dummies = pd.concat(dummies, axis=1)
bids_train_dummies.to_csv('bids_train_dummies.csv')

dummies = []
for key in bids_test_small.keys():
    dummies.append(pd.get_dummies(bids_test_small[key]))
bids_test_dummies = pd.concat(dummies, axis=1)
bids_test_dummies.to_csv('bids_test_dummies.csv')

# SGD (stochstic gradient descent classifier) with logloss
# from sklearn.linear_model import SGDClassifier

# clf = SGDClassifier(loss="log", verbose=1, random_state=1234)
# clf.fit(bids_train_dummies, outcome)
# >>> clf.predict_proba([[1., 1.]])


# label encoding - dummy variables?
# ref: http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
# http://stackoverflow.com/questions/25530504/encoding-column-labels-in-pandas-for-machine-learning

# large number of sparse labels classification
# http://research.microsoft.com/en-us/um/people/manik/pubs%5Cagrawal13.pdf
# http://jmlr.org/proceedings/papers/v32/yu14.pdf

# Idea 1: classify the bidder by useful-looking labels: auction,
# merchandise, device, time, country, url

# Idea 2: cluster the bids by their information and use them as a
# classifier?
