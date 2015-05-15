import pandas as pd
import numpy as np

bids_test = pd.read_csv('data/bids_test.csv')

def get_max_auc(bids):

    bidder_id = bids['bidder_id'].unique()
    num_bidders = len(bidder_id)
    
    # get the maximum number of auctions participated by one bidder
    auclen = []
    for i in range(num_bidders):
        print "%d/%d" %(i, num_bidders)
        
        bbbidder = bids[bids['bidder_id'] == bidder_id[i]]
        nbfea = []
        # count number of bids for each auction
        for auc in bbbidder['auction'].unique():
            nbfea.append(len(bbbidder[bbbidder['auction'] == auc]))

        auclen.append(len(nbfea))
    print max(auclen)

    return num_bidders, max(auclen)
    
num_bidders, maxa = get_max_auc(bids_test)


def gather_info(num_bidders, max_auc, max_auc_count, bids, class_id):
    """
    Gather the useful infromation from bids data.
    """
    
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
