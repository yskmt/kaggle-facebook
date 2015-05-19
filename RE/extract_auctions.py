"""
extract_auctions.py
Facebook Recruiting IV: Human or Robot?
author: Yusuke Sakamoto

Extract each auctions as a part of preprocessing
"""

import numpy as np
import pandas as pd

print "Loading human/bots/test data..."
bids_humans = pd.read_csv('data/bids_humans.csv')
bids_bots = pd.read_csv('data/bids_bots.csv')
bids_test = pd.read_csv('data/bids_test.csv')

aucs_human = bids_humans['auction'].unique()
aucs_bot = bids_bots['auction'].unique()
aucs_test = bids_test['auction'].unique()

print "gather all the auctions participated by human, bots, and individuals"\
    " in test set."
auctions = set(aucs_human).union(set(aucs_bot)).union(set(aucs_test))
auctions = list(auctions)
num_auctions = len(auctions)

print "Loading bids data..."
bidsdata = pd.read_csv('data/bids.csv')

print "Gather all the bids in the auctinos we are interested."
# save at each 1000 iterations for memory saving purpose...
num_chunk = 1000
num_itr = num_auctions/num_chunk+1
act = 0
for k in range(num_itr):
    aucs_bids = []
    num_auc_itr = min(num_chunk, num_auctions-act)
    
    for ct in range(num_auc_itr): 
        auc = auctions[act]

        if act%100 == 0:
            print "%d/%d" %(act, num_aucs)
            
        aucs_bids.append(
            bidsdata[bidsdata.auction==auc]
        )
        act += 1

    aucs_bids = pd.concat(aucs_bids, axis=0)
    aucs_bids.to_csv('data/aucs_bids_%000d.csv' %k)

# cleanup the csv files
aucsbids = []
for i in range(16):
    aucsbids.append(pd.read_csv('data/aucs_bids_%d.csv' %i))

ab = pd.concat(aucsbids, axis=0)
ab.drop(['Unnamed: 0'], inplace=True, axis=1)

ab.to_csv('data/aucs_bids.csv', index_label='idx')

