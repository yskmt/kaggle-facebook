"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd


# maximum number of auction particpated by one bot == 1018
# NOTE!! for number of bids for each auction data (sorted in descendent order),
# I should converge all the bids after a sufficient number of auctions,
# as most of the bidders only bid less than 500 times. Or categorize them
# into 1~10 bids, 11~100 bids, etc. The criterial should be chosen by
# comparing bids by bots and human

# maximum number of auctions participated by one [human, bot, test]
max_auc = [1623, 1018, 1726]
max_auc_count = 100


def gather_info(num_bidders, max_auc, max_auc_count, bids, class_id):

    # ANALYSIS
    tmp_auc = np.zeros((num_bidders, max_auc), dtype=int)
    tmp = np.zeros((num_bidders, 3), dtype=int)
    tmp_mch = np.zeros(num_bidders, dtype=object)

    # for each bidder
    for i in range(num_bidders):
        if i%10 == 0:
            print "%d/%d" %(i, num_bidders)
        # bids by this bidder
        bbbidder = bids[bids['bidder_id'] == class_id[i]]

        # number of bids by this bidder
        num_bbbidder = len(bbbidder)
        # number of auction by this bidder
        num_abbidder = len(bbbidder['auction'].unique())

        # count number of bids for each auction
        nbfea = []
        for auc in bbbidder['auction'].unique():
            nbfea.append(len(bbbidder[bbbidder['auction'] == auc]))

        tmp_auc[i, :len(nbfea)] = sorted(nbfea, reverse=True)
        # NOTE: each bidder only has ONE unique merchandise, check
        # num_merchandise attribute
        tmp_mch[i] = bbbidder['merchandise'].unique()[0]

        tmp[i, 0] = num_bbbidder
        tmp[i, 1] = num_abbidder
        tmp[i, 2] = len(bbbidder['merchandise'].unique())

    bidders_mch = pd.get_dummies(pd.DataFrame(tmp_mch, index=class_id,
                                              columns=['merchandise']))
    bidders_info = pd.DataFrame(tmp, index=class_id,
                                columns=list(['num_bids',
                                              'num_aucs',
                                              'num_merchandise']))
    bidders_bids_by_aucs = pd.DataFrame(
        tmp_auc, index=class_id,
        columns=map(lambda x: 'num_bids_by_auc_'+str(x), range(max_auc)))

    bidders_info = pd.concat([bidders_info, bidders_mch,
                              bidders_bids_by_aucs.iloc[:, :max_auc_count]],
                             axis=1)

    return bidders_info, bidders_bids_by_aucs


# print "Loading bids data..."
# bids_bots = pd.read_csv('data/bids_bots.csv')
# bids_human = pd.read_csv('data/bids_human.csv')

# bots_id = (bids_bots['bidder_id']).unique()
# human_id = (bids_human['bidder_id']).unique()

# num_bots = len(bots_id)
# num_human = len(human_id)

bids_test = pd.read_csv('data/bids_test.csv')
test_id = bids_test['bidder_id'].unique()
num_test = len(test_id)

# print "Analysing bots data..."

# bots_info, bots_bids_by_aucs\
#     = gather_info(num_bots, max_auc[1], max_auc_count, bids_bots, bots_id)

# bots_info.to_csv('data/bots_info.csv', index_label='bidder_id')
# bots_bids_by_aucs.to_csv('data/bots_bids_by_aucs.csv', index_label='bidder_id')

# print "Analysing huaman data..."

# human_info, human_bids_by_aucs\
#     = gather_info(num_human, max_auc[0], max_auc_count, bids_human, human_id)

# human_info.to_csv('data/human_info.csv', index_label='bidder_id')
# human_bids_by_aucs.to_csv('data/human_bids_by_aucs.csv', index_label='bidder_id')

print "Analysing test data..."

test_info, test_bids_by_aucs\
    = gather_info(num_test, max_auc[2], max_auc_count, bids_test, test_id)

test_info.to_csv('data/test_info.csv', index_label='bidder_id')
test_bids_by_aucs.to_csv('data/test_bids_by_aucs.csv', index_label='bidder_id')
