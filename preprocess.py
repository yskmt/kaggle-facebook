# """
# preprocess.py

# Facebook Recruiting IV: Human or Robot?

# author: Yusuke Sakamoto

# """

# import numpy as np
# import pandas as pd


# trainfile = "data/train.csv"
# testfile = "data/test.csv"
# bidsfile = "data/bids.csv"

# traindata = pd.read_csv(trainfile)
# bidsdata = pd.read_csv(bidsfile)
# testdata = pd.read_csv(testfile)

# botdata = traindata[traindata.outcome == 1]
# humandata = traindata[traindata.outcome == 0]

# num_bots = botdata.shape[0]

# print "collecting the bids by bots..."
# bids_bots = []
# for i in range(num_bots):
#     bids_bots.append(bidsdata[bidsdata.bidder_id == botdata.iloc[i, 0]])
# bids_bots = pd.concat(bids_bots)

# print "collecting the bids by human..."
# bids_human = []
# num_humans = humandata.shape[0]
# for j in range(num_humans):
#     bids_human.append(bidsdata[bidsdata.bidder_id == humandata.iloc[j, 0]])
# bids_human = pd.concat(bids_human)

# print "collecting the bids from test data (not sure human or bots)..."
# bids_test = []
# num_test = testdata.shape[0]
# for j in range(num_test):
#     bids_test.append(bidsdata[bidsdata.bidder_id == testdata.iloc[j, 0]])
# bids_test = pd.concat(bids_test)

# bids_bots = pd.read_csv('data/bids_bots.csv')
# bids_human = pd.read_csv('data/bids_human.csv')
# bids_test = pd.read_csv('data/bids_test.csv')

# print "dropping unnecessary columns..."
# bids_bots.drop('Unnamed: 0', axis=1, inplace=True)
# bids_human.drop('Unnamed: 0', axis=1, inplace=True)
# bids_test.drop('Unnamed: 0', axis=1, inplace=True)

# print "saving bids data..."
# bids_bots.to_csv('data/bids_bots.csv', index=False)
# bids_human.to_csv('data/bids_human.csv', index=False)
# bids_test.to_csv('data/bids_test.csv', index=False)

# bids_bots = pd.read_csv('bids_bots.csv')
# bids_human = pd.read_csv('bids_human.csv')
# bids_test = pd.read_csv('bids_test.csv')

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


############################################################################
## append/drop some labels and save

# print "Loading postprocessed data files..."

# humanfile = 'data/human_info.csv'
# botsfile = 'data/bots_info.csv'
# testfile = 'data/test_info.csv'

# human_info = pd.read_csv(humanfile, index_col=0)
# bots_info = pd.read_csv(botsfile, index_col=0)
# test_info = pd.read_csv(testfile, index_col=0)

# drop num_bids_by_auc_*
# for key in human_info.keys():
#     if 'num_bids_by_auc' in key:
#         human_info.drop(key, inplace=True, axis=1)
    
# for key in bots_info.keys():
#     if 'num_bids_by_auc' in key:
#         bots_info.drop(key, inplace=True, axis=1)

        
# for key in test_info.keys():
#     if 'num_bids_by_auc' in key:
#         test_info.drop(key, inplace=True, axis=1)

# num_human = human_info.shape[0]
# num_bots = bots_info.shape[0]
# num_test = test_info.shape[0]

# # number of bids per auction data (descending)
# bbba = pd.read_csv('data/bots_bids_by_aucs.csv', index_col=0)
# hbba = pd.read_csv('data/human_bids_by_aucs.csv', index_col=0)
# tbba = pd.read_csv('data/test_bids_by_aucs.csv', index_col=0)

# # TODO: save these empty data in _info.csv files from the beginning
# # merchandise that do not belong to bots
# extra_merchandise = ['merchandise_auto parts', 'merchandise_clothing',
#                      'merchandise_furniture']
# bots_info_append = pd.DataFrame(
#     np.zeros((num_bots, 3)), columns=extra_merchandise, index=bots_info.index)
# bots_info = pd.concat([bots_info, bots_info_append], axis=1)
# # train_info = pd.concat([human_info, bots_info], axis=0)

# # merchandise that do not belong to test
# extra_merchandise = ['merchandise_auto parts']
# test_info_append = pd.DataFrame(
#     np.zeros((num_test, 1)), columns=extra_merchandise, index=test_info.index)
# test_info = pd.concat([test_info, test_info_append], axis=1).sort(axis=1)


# human_info.to_csv(humanfile, index_col=0)
# bots_info.to_csv(botsfile, index_col=0)
# test_info.to_csv(testfile, index_col=0)
