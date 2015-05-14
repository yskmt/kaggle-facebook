"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd

# print "saving bids data..."
bids_bots = pd.read_csv('data/bids_bots.csv')
bids_human = pd.read_csv('data/bids_human.csv')

bots_id = (bids_bots['bidder_id']).unique()
human_id = (bids_human['bidder_id']).unique()

num_bots = len(bots_id)
num_human = len(human_id)


# maximum number of auction particpated by one bot == 1018
# NOTE!! for number of bids for each auction data (sorted in descendent order),
# I should converge all the bids after a sufficient number of auctions,
# as most of the bidders only bid less than 500 times. Or categorize them
# into 1~10 bids, 11~100 bids, etc. The criterial should be chosen by
# comparing bids by bots and human

# maximum number of auctions participated by one [human, bot]
max_auc = [1623, 1018]

# ANALYSIS
tmpb_auc = np.zeros((num_bots, max_auc[1]), dtype=int)
tmpb = np.zeros((num_bots, 3), dtype=int)
tembp_mch = np.zeros(num_bots, dtype=object)

for i in range(num_bots):
    # bids by this bot
    bbbots = bids_bots[bids_bots['bidder_id'] == bots_id[i]]

    # number of bids by this bot
    num_bbbots = len(bbbots)
    # number of auction by this bot
    num_abbots = len(bbbots['auction'].unique())
    
    nbfea = []
    # count number of bids for each auction
    for auc in bbbots['auction'].unique():
        nbfea.append(len(bbbots[bbbots['auction'] == auc]))

    tmpb_auc[i, :len(nbfea)] = sorted(nbfea, reverse=True)

    tmpb[i, 0] = num_bbbots
    tmpb[i, 1] = num_abbots
    tmpb[i, 2] = num_merchandise
    tempb_mch[i] = bbbots['merchandise'].unique()

bots_mch = pd.DataFrame(tempb_mch)
bots_info = pd.DataFrame(tmpb, index=bots_id, columns=list(['num_bids', 'num_aucs', 'num_merchandise']))
bots_bids_by_aucs = pd.DataFrame(tmpb_auc, index=bots_id)

bots_info.to_csv('data/bots_info.csv', index_label='bidder_id')
bots_bids_by_aucs.to_csv('data/bots_bids_by_aucs.csv', index_label='bidder_id')

# HUMAN
tmph_auc = np.zeros((num_human, max_auc[0]), dtype=int)
tmph = np.zeros((num_human, 3), dtype=int)
    
for i in range(num_human):
    # bids by this bot
    bbhuman = bids_human[bids_human['bidder_id'] == human_id[i]]

    # number of bids by this bot
    num_bbhuman = len(bbhuman)
    # number of auction by this bot
    num_abhuman = len(bbhuman['auction'].unique())
    num_merchandise = len(bbbots['merchandise'].unique())
    
    nbfea = []
    # count number of bids for each auction
    for auc in bbhuman['auction'].unique():
        nbfea.append(len(bbhuman[bbhuman['auction'] == auc]))

    tmph_auc[i, :len(nbfea)] = sorted(nbfea, reverse=True)

    tmph[i, 0] = num_bbhuman
    tmph[i, 1] = num_abhuman
    tmph[i, 2] = num_merchandise
    
human_info = pd.DataFrame(tmph, index=human_id, columns=list(['num_bids', 'num_aucs', 'num_merchandise']))
human_bids_by_aucs = pd.DataFrame(tmph_auc, index=human_id)

human_info.to_csv('data/human_info.csv', index_label='bidder_id')
human_bids_by_aucs.to_csv('data/human_bids_by_aucs.csv', index_label='bidder_id')
