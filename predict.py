"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd


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
# humans, under-sample the human data
# num_human_ex = min(num_bots, num_human)
# index_shuffle = range(num_human_ex)
# np.random.shuffle(index_shuffle)
# X_human = X_train[index_shuffle]
# y = np.concatenate([y[:num_human_ex], y[num_human:]], axis=0)
# X_train = np.concatenate([X_human, X_train[num_human:, :]], axis=0)
# shuffle again just in case
# index_shuffle2 = range(len(y))
# np.random.shuffle(index_shuffle2)
# y = y[index_shuffle2]
# X_train = X_train[index_shuffle2]

# over-sample
multiplicity = num_human/num_bots
bots_info_os = [bots_info] * multiplicity
train_info = pd.concat([human_info] + bots_info_os, axis=0).sort(axis=1)
X_train = train_info.values[:, 1:]
y = np.concatenate([np.zeros(num_human), np.ones(num_bots*multiplicity)], axis=0)
# shuffle!
index_shuffle = range(len(y))
np.random.shuffle(index_shuffle)
X_train = X_train[index_shuffle]
y = y[index_shuffle]

X_test = test_info.values[:, 1:]

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, n_jobs=2, random_state=1234, verbose=1,
                             max_features='auto')
# clf = SGDClassifier(loss="log", verbose=1, random_state=1234, n_iter=5000)
clf.fit(X_train, y)

# prediction on test set
y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# measuring prediction peformance agianst train set
train_proba = clf.predict_proba(X_train)
train_pred = clf.predict(X_train)

### 70 bidders in test.csv do not have any data in bids.csv. Thus they
### are not included in analysis/prediction, but they need to be
### appended in the submission. The prediction of these bidders do not matter.

test_ids_all = pd.read_csv('data/test.csv')['bidder_id']
test_ids_append = list(set(test_ids_all.values).difference(set(test_ids.values)))
submission_append = pd.DataFrame(np.zeros(len(test_ids_append)),
                                          index=test_ids_append, columns=['prediction'])

# Make as submission file!
submission = pd.DataFrame(y_proba[:,1], index=test_ids, columns=['prediction'])
submission = pd.concat([submission, submission_append], axis=0)
submission.to_csv('data/submission.csv', index_label='bidder_id')
