
# get the maximum number of auctions participated by one bidder
auclen = []
for i in range(num_human):
    bbhuman = bids_human[bids_human['bidder_id'] == human_id[i]]
    nbfea = []
    # count number of bids for each auction
    for auc in bbhuman['auction'].unique():
        nbfea.append(len(bbhuman[bbhuman['auction'] == auc]))

    auclen.append(len(nbfea))
print max(auclen)
