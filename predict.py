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

start_time = time.time()
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

# take only some data from auc_count
max_auc_count = 1000
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


# drop num item labels
# columns_dropped = [u'num_merchandise', u'num_devices', u'num_countries', u'num_ips',
                   # u'num_urls']
columns_dropped = [u'num_merchandise']
human_info.drop(columns_dropped, axis=1, inplace=True)
bots_info.drop(columns_dropped, axis=1, inplace=True)
test_info.drop(columns_dropped, axis=1, inplace=True)

# drop merchandise dummy variables
for key in human_info.keys():
    if 'merchandise' in key:
        human_info.drop([key], axis=1, inplace=True)
        bots_info.drop([key], axis=1, inplace=True)
        test_info.drop([key], axis=1, inplace=True)

human_info.sort(axis=1)
bots_info.sort(axis=1)
test_info.sort(axis=1)


############################################################################
# Predicting: Random Forest with bagging
############################################################################

# bagging with bootstrap
y_valids = []
roc_auc_mean = []
roc_auc_std = []
bots_rate_mean = []
bots_rate_std = []
specificity_mean = []
specificity_std = []
score_mean = []
score_std = []
holdout = 0.2
# ks = range(1,18,4)
ks = range(20, 21)
for k in ks:
    num_sim = 5
    y_probas = []
    roc_aucs = []
    tprs = []
    scovs = []
    for i in range(num_sim):
        np.random.seed(int(time.time() * 1000 % 4294967295))

        y_proba, y_pred, train_proba, train_pred, roc_auc, tpr, scov\
            = predict_usample(num_human, num_bots, human_info,
                              bots_info, test_info, holdout=holdout,
                              multiplicity=k, plot_roc=True)

        y_probas.append(y_proba[:, 1])  # gather the bot probabilities
        roc_aucs.append(roc_auc)
        tprs.append(tpr)
        scovs.append(scov)

    tprs = np.array(tprs)
    specificity_mean.append(tprs.mean())
    specificity_std.append(tprs.std())

    scovs = np.array(scovs)
    score_mean.append(scovs.mean())
    score_std.append(scovs.std())

    # postprocessing
    y_probas = np.array(y_probas)
    y_proba_ave = y_probas.T.mean(axis=1)

    bots_rates = np.sum(y_probas > 0.5, axis=1) / \
        np.array(map(len, y_probas), dtype=float)
    bots_rate_mean.append(bots_rates.mean())
    bots_rate_std.append(bots_rates.std())

    roc_aucs = np.array(roc_aucs)
    roc_auc_mean.append(roc_aucs.mean())
    roc_auc_std.append(roc_aucs.std())

    # print "k: ", k
    # print "bots proba for test set: ", brs.mean()

np.set_printoptions(suppress=True, precision=3)
print "CV result:"
cv_result = np.round(np.array([ks[0], roc_auc_mean[0], roc_auc_std[0],
                               bots_rate_mean[0], bots_rate_std[0],
                               specificity_mean[0],
                               specificity_std[0], score_mean[0],
                               score_std[0]]), 3)
cv_scores = ['multiplicity', 'roc_auc: mean', 'roc_auc: std',
             'bots_rate: mean', 'bote_rate: std', 'specificity: mean',
             'specificity: std', 'accuracy: mean', 'accuracy: std']

print pd.DataFrame(cv_result, index=cv_scores)

# 70 bidders in test.csv do not have any data in bids.csv. Thus they
# are not included in analysis/prediction, but they need to be
# appended in the submission. The prediction of these bidders do not matter.

test_ids_all = pd.read_csv('data/test.csv')['bidder_id']
test_ids_append = list(
    set(test_ids_all.values).difference(set(test_ids.values)))
submission_append = pd.DataFrame(np.zeros(len(test_ids_append)),
                                 index=test_ids_append, columns=['prediction'])

# Make as submission file!
submission = pd.DataFrame(y_proba_ave, index=test_ids, columns=['prediction'])
submission = pd.concat([submission, submission_append], axis=0)
submission.to_csv('data/submission.csv', index_label='bidder_id')

print "bots proba for train set:", num_bots / float(num_human + num_bots)
print "bots proba for test set: ", sum(y_proba_ave > 0.5) / float(len(y_proba_ave))

end_time = time.time()

print "Time elapsed: %.2f" %(end_time-start_time)
