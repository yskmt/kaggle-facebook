"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
import time

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


# number of bids per auction data (descending)
bbba = pd.read_csv('data/bots_bids_by_aucs.csv')
hbba = pd.read_csv('data/human_bids_by_aucs.csv')
tbba = pd.read_csv('data/test_bids_by_aucs.csv')
# take the minimum number of auction bidded
max_auc_count = 4
max_auc_count = min([bbba.shape[1], hbba.shape[1], tbba.shape[1],
                     max_auc_count])


## TODO: save these empty data in _info.csv files from the beginning
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

############################################################################
# Data dropping/appending
############################################################################

## TODO: just don't save the num_bids_by_auc_ data in _info
## files. They are saved in _bids_by_aucs.csv files anyway.

# # drop num_bids_by_auc_ columns
# for i in range(0,100):
#     human_info.drop(['num_bids_by_auc_%d' %i], axis=1, inplace=True)
#     bots_info.drop(['num_bids_by_auc_%d' %i], axis=1, inplace=True)
#     test_info.drop(['num_bids_by_auc_%d' %i], axis=1, inplace=True)
if max_auc_count>0:
    for key in human_info.keys():
        if 'num_bids_by_auc' in key:
            human_info.drop([key], axis=1, inplace=True)
            bots_info.drop([key], axis=1, inplace=True)
            test_info.drop([key], axis=1, inplace=True)

    # append num_bids_by_auc_ columns
    # max_auc_count = 100
    hbba.fillna(0)
    bbba.fillna(0)
    tbba.fillna(0)
    human_info = pd.concat([human_info, hbba.iloc[:, 1:max_auc_count]], axis=1)
    bots_info = pd.concat([bots_info,  bbba.iloc[:, 1:max_auc_count]], axis=1)
    test_info = pd.concat([test_info,  tbba.iloc[:, 1:max_auc_count]], axis=1)
    
# columns_dropped = [u'num_merchandise', u'num_devices', u'num_countries', u'num_ips', u'num_urls']
columns_dropped = [u'num_merchandise']
human_info.drop(columns_dropped, axis=1, inplace=True)
bots_info.drop(columns_dropped, axis=1, inplace=True)
test_info.drop(columns_dropped, axis=1, inplace=True)

# merchandise dummy variables
# for key in human_info.keys():
#     if 'merchandise' in key:
#         human_info.drop([key], axis=1, inplace=True)
#         bots_info.drop([key], axis=1, inplace=True)
#         test_info.drop([key], axis=1, inplace=True)

human_info.sort(axis=1)
bots_info.sort(axis=1)
test_info.sort(axis=1)


# bagging with bootstrap
vs_cv = []
for k in range(20, 21):
    num_sim = 10
    y_probas = []
    valid_score = 0
    for i in range(num_sim):
        np.random.seed(int(time.time()*1000%4294967295))
        y_proba, y_pred, train_proba, train_pred, auc_valid \
            = predict_usample(num_human, num_bots, human_info,
                              bots_info, test_info, holdout=0.,
                              multiplicity=k)
        y_probas.append(y_proba[:,1])  # gather the bot probabilities
        valid_score += auc_valid

    # print "valid score: ", valid_score/float(num_sim)
    vs_cv.append(valid_score/float(num_sim))

print "CV result:"
print vs_cv
    
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



print "bots proba for train set:", num_bots/float(num_human+num_bots)
print "bots proba for test set: ", sum(y_proba_ave>0.5)/float(len(y_proba_ave))
