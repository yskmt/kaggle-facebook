"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd

from etc import gather_info

# maximum number of auction particpated by one bot == 1018
# NOTE!! for number of bids for each auction data (sorted in descendent order),
# I should converge all the bids after a sufficient number of auctions,
# as most of the bidders only bid less than 500 times. Or categorize them
# into 1~10 bids, 11~100 bids, etc. The criterial should be chosen by
# comparing bids by bots and human

# maximum number of auctions participated by one [human, bot, test]
max_auc = [1623, 1018, 1726]
max_auc_count = 100  # maximum auction count to analyze

print "Loading bids data..."
bids_bots = pd.read_csv('data/bids_bots.csv')
bids_human = pd.read_csv('data/bids_human.csv')
bids_test = pd.read_csv('data/bids_test.csv')

bots_id = (bids_bots['bidder_id']).unique()
human_id = (bids_human['bidder_id']).unique()
test_id = bids_test['bidder_id'].unique()

num_bots = len(bots_id)
num_human = len(human_id)
num_test = len(test_id)

print "Analyzing huaman data..."
human_info, human_bids_by_aucs\
    = gather_info(num_human, max_auc[0], max_auc_count, bids_human, human_id)
human_info.to_csv('data/human_info.csv', index_label='bidder_id')
human_bids_by_aucs.to_csv('data/human_bids_by_aucs.csv', index_label='bidder_id')

print "Analyzing bots data..."
bots_info, bots_bids_by_aucs\
    = gather_info(num_bots, max_auc[1], max_auc_count, bids_bots, bots_id)
bots_info.to_csv('data/bots_info.csv', index_label='bidder_id')
bots_bids_by_aucs.to_csv('data/bots_bids_by_aucs.csv', index_label='bidder_id')

print "Analyzing test data..."
test_info, test_bids_by_aucs\
    = gather_info(num_test, max_auc[2], max_auc_count, bids_test, test_id)
test_info.to_csv('data/test_info.csv', index_label='bidder_id')
test_bids_by_aucs.to_csv('data/test_bids_by_aucs.csv', index_label='bidder_id')

