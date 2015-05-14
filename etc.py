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
