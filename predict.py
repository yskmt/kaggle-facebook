"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from etc import predict_usample

print "Loading postprocessed data files..."

humanfile = 'data/human_info.csv'
botsfile = 'data/bots_info.csv'
testfile = 'data/test_info.csv'

human_info = pd.read_csv(humanfile, index_col=0)
bots_info = pd.read_csv(botsfile, index_col=0)
test_info = pd.read_csv(testfile, index_col=0)

num_human = human_info.shape[0]
num_bots = bots_info.shape[0]
num_test = test_info.shape[0]

# number of bids per auction data (descending)
bbba = pd.read_csv('data/bots_bids_by_aucs.csv', index_col=0)
hbba = pd.read_csv('data/human_bids_by_aucs.csv', index_col=0)
tbba = pd.read_csv('data/test_bids_by_aucs.csv', index_col=0)

# take the minimum number of auction bidded
max_auc_count = 100
max_auc_count = min([bbba.shape[1], hbba.shape[1], tbba.shape[1],
                     max_auc_count])


####
# train_ids = train_info['bidder_id']
test_ids = test_info.index

############################################################################
# Data dropping/appending
############################################################################

# append num_bids_by_auc_ columns
if max_auc_count > 0:
    # max_auc_count = 100
    hbba.fillna(0)
    bbba.fillna(0)
    tbba.fillna(0)
    human_info = pd.concat([human_info, hbba.iloc[:, :max_auc_count]], axis=1)
    bots_info = pd.concat([bots_info,  bbba.iloc[:, :max_auc_count]], axis=1)
    test_info = pd.concat([test_info,  tbba.iloc[:, :max_auc_count]], axis=1)


# drop: [u'num_merchandise', u'num_devices', u'num_countries', u'num_ips',
# u'num_urls']
columns_dropped = [u'num_merchandise']
human_info.drop(columns_dropped, axis=1, inplace=True)
bots_info.drop(columns_dropped, axis=1, inplace=True)
test_info.drop(columns_dropped, axis=1, inplace=True)

# drop merchandise dummy variables
# for key in human_info.keys():
#     if 'merchandise' in key:
#         human_info.drop([key], axis=1, inplace=True)
#         bots_info.drop([key], axis=1, inplace=True)
#         test_info.drop([key], axis=1, inplace=True)

human_info.sort(axis=1)
bots_info.sort(axis=1)
test_info.sort(axis=1)

human_data = human_info.values
bots_data = bots_info.values
test_data = test_info.values

train_data = np.vstack((human_data, bots_data))
y_train = np.hstack((np.zeros(num_human), np.ones(num_bots)))

num_train = train_data.shape[0]

ins = range(num_train)
np.random.shuffle(ins)

X_train = train_data[ins, :]
y_train = y_train[ins]

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clf = RandomForestClassifier(n_estimators=100, n_jobs=2,
                                 random_state=1234, verbose=0,
                                 max_features='auto')

ntr = int(num_train*0.8)
nts = num_train-ntr

clf.fit(X_train[:ntr, :], y_train[:ntr])
y_proba = clf.predict_proba(X_train[ntr:, :])
y_pred  = clf.predict(X_train[ntr:, :])
y_valid = y_train[ntr:]

fpr, tpr, thresholds = roc_curve(y_train[ntr:], y_proba[:, 1])
roc_auc = auc(fpr, tpr)


def specificity(y_true, y_pred):

    positive = sum(y_true)
    true_positive = sum(np.logical_and(y_pred==1, y_valid==1))

    return true_positive/float(positive)
    
# # bagging with bootstrap
# score_cv = []
# std_cv = []
# br_mean = []
# br_std = []
# y_valids = []
# holdout = 0.0
# # ks = range(1,18,4)
# ks = range(5, 9)
# # for k in range(1, 18, 4):
# for k in ks:
#     num_sim = 25
#     y_probas = []
#     ras = []
#     for i in range(num_sim):
#         np.random.seed(int(time.time() * 1000 % 4294967295))
#         y_proba, y_pred, train_proba, train_pred, roc_auc \
#             = predict_usample(num_human, num_bots, human_info,
#                               bots_info, test_info, holdout=holdout,
#                               multiplicity=k)
#         y_probas.append(y_proba[:, 1])  # gather the bot probabilities
#         ras.append(roc_auc)

#     # postprocessing
#     y_probas = np.array(y_probas)
#     y_proba_ave = y_probas.T.mean(axis=1)

#     brs = np.sum(y_probas > 0.5, axis=1) / \
#         np.array(map(len, y_probas), dtype=float)
#     br_mean.append(brs.mean())
#     br_std.append(brs.std())

#     ras = np.array(ras)
#     score_cv.append(ras.mean())
#     std_cv.append(ras.std())

#     print "k: ", k
#     print "bots proba for test set: ", brs.mean()

# np.set_printoptions(suppress=True, precision=3)
# print "CV result:"
# print np.round(np.array([ks, score_cv, std_cv, br_mean, br_std]), 3)


# # 70 bidders in test.csv do not have any data in bids.csv. Thus they
# # are not included in analysis/prediction, but they need to be
# # appended in the submission. The prediction of these bidders do not matter.

# test_ids_all = pd.read_csv('data/test.csv')['bidder_id']
# test_ids_append = list(
#     set(test_ids_all.values).difference(set(test_ids.values)))
# submission_append = pd.DataFrame(np.zeros(len(test_ids_append)),
#                                  index=test_ids_append, columns=['prediction'])

# # Make as submission file!
# submission = pd.DataFrame(y_proba_ave, index=test_ids, columns=['prediction'])
# submission = pd.concat([submission, submission_append], axis=0)
# submission.to_csv('data/submission.csv', index_label='bidder_id')

# print "bots proba for train set:", num_bots / float(num_human + num_bots)
# print "bots proba for test set: ", sum(y_proba_ave > 0.5) / float(len(y_proba_ave))
