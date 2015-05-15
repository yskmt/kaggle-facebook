"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd

from etc import predict_usample

print "Loading postprocessed data files..."

humanfile = 'data/human_info.csv'
botsfile = 'data/bots_info.csv'
testfile = 'data/test_info.csv'

human_info = pd.read_csv(humanfile)
bots_info = pd.read_csv(botsfile)
test_info = pd.read_csv(testfile)

num_human = human_info.shape[0]
num_bots = bots_info.shape[0]
num_test = test_info.shape[0]
num_train = num_human + num_bots

# merchandise that do not belong to bots
extra_merchandise = ['merchandise_auto parts', 'merchandise_clothing',
                     'merchandise_furniture']
bots_info_append = pd.DataFrame(np.zeros((num_bots, 3)), columns=extra_merchandise)
bots_info = pd.concat([bots_info, bots_info_append], axis=1)
# train_info = pd.concat([human_info, bots_info], axis=0)

# merchandise that do not belong to test
extra_merchandise = ['merchandise_auto parts']
test_info_append = pd.DataFrame(np.zeros((num_test, 1)), columns=extra_merchandise)
test_info = pd.concat([test_info, test_info_append], axis=1).sort(axis=1)

####
# train_ids = train_info['bidder_id']
test_ids = test_info['bidder_id']

# y = np.concatenate([np.zeros(num_human), np.ones(num_bots)], axis=0)
# X_train = train_info.values[:, 1:]
# X_test = test_info.values[:, 1:]

# index_shuffle = range(num_train)
# np.random.shuffle(index_shuffle)
# y = y[index_shuffle]
# X_train = X_train[index_shuffle, :]

# Because number of bots is significantly smaller than number of
# humans, special care needs to be taken

# # over-sample the bots data
# multiplicity = num_human/num_bots
# bots_info_os = [bots_info] * multiplicity
# train_info = pd.concat([human_info] + bots_info_os, axis=0).sort(axis=1)
# X_train = train_info.values[:, 1:]
# y = np.concatenate([np.zeros(num_human), np.ones(num_bots*multiplicity)], axis=0)

# drop unnecessary columns
for i in range(100):
    human_info.drop(['num_bids_by_auc_%d' %i], axis=1, inplace=True)
    bots_info.drop(['num_bids_by_auc_%d' %i], axis=1, inplace=True)
    test_info.drop(['num_bids_by_auc_%d' %i], axis=1, inplace=True)

# columns_dropped = [u'num_merchandise', u'num_devices', u'num_countries', u'num_ips', u'num_urls']
columns_dropped = [u'num_merchandise']
human_info.drop(columns_dropped, axis=1, inplace=True)
bots_info.drop(columns_dropped, axis=1, inplace=True)
test_info.drop(columns_dropped, axis=1, inplace=True)

# for key in human_info.keys():
#     if 'merchandise' in key:
#         human_info.drop([key], axis=1, inplace=True)
#         bots_info.drop([key], axis=1, inplace=True)
#         test_info.drop([key], axis=1, inplace=True)

# bagging with bootstrap
y_probas = []
for i in range(1):
    y_proba, y_pred, train_proba, train_pred \
        = predict_usample(num_human, num_bots, human_info, bots_info, test_info)
    y_probas.append(y_proba[:,1])  # gather the bot probabilities

y_probas = np.array(y_probas)
y_proba_ave = y_probas.T.mean(axis=1)
    
### 70 bidders in test.csv do not have any data in bids.csv. Thus they
### are not included in analysis/prediction, but they need to be
### appended in the submission. The prediction of these bidders do not matter.

test_ids_all = pd.read_csv('data/test.csv')['bidder_id']
test_ids_append = list(set(test_ids_all.values).difference(set(test_ids.values)))
submission_append = pd.DataFrame(np.zeros(len(test_ids_append)),
                                          index=test_ids_append, columns=['prediction'])

# Make as submission file!
submission = pd.DataFrame(y_proba_ave, index=test_ids, columns=['prediction'])
submission = pd.concat([submission, submission_append], axis=0)
submission.to_csv('data/submission.csv', index_label='bidder_id')

print sum(y_pred)/float(len(y_proba_ave))
