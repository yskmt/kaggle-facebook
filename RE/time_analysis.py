# """
# preprocess.py

# Facebook Recruiting IV: Human or Robot?

# author: Yusuke Sakamoto

# """

import numpy as np
import pandas as pd


# bids_bots = pd.read_csv('data/bids_bots.csv')
# info_bots = pd.read_csv('data/info_bots.csv', index_col=0)

# bidstime = bids_bots[bids_bots['bidder_id']==info_bots.index[1]]['time'].values
# bidstime -= bidstime[0]
# bidstime /= 1e10

# print bidstime[0:-1]


bids_humans = pd.read_csv('data/bids_humans.csv')
bids_bots = pd.read_csv('data/bids_bots.csv')

info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
info_bots = pd.read_csv('data/info_bots.csv', index_col=0)

ih = info_humans[info_humans.num_bids>100]
for idx in ih.index:
    print idx
    bh = bids_humans[bids_humans['bidder_id']==idx]
    bidstime = bh['time'].values
    bidstime -= bidstime[0]
    bidstime /= 1e10

    if len(bidstime)>100:
        print bidstime


ib = info_bots[info_bots.num_bids>100]
for idx in ib.index:
    print idx
    bh = bids_bots[bids_bots['bidder_id']==idx]
    bidstime = bh['time'].values
    bidstime -= bidstime[0]
    bidstime /= 1e10

    if len(bidstime)>100:
        print bidstime



# numb = pd.DataFrame(
#     {'bots': info_bots.num_bids.values,
#      'humans': info_humans.num_bids.values}, columns=['bots', 'humans'])

# numb.plot(kind='hist')




# plot distributions of num_*

import scipy.stats

_, ld = scipy.stats.boxcox(info_bots.num_ips)
pd.DataFrame(scipy.stats.boxcox(info_bots.num_ips, ld)).plot(kind='hist', alpha=0.5)
pd.DataFrame(scipy.stats.boxcox(info_humans.num_ips, ld)).plot(kind='hist', alpha=0.5)
pd.DataFrame(scipy.stats.boxcox(info_test.num_ips, ld)).plot(kind='hist', alpha=0.5)
plt.show()


