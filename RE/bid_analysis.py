"""
time_analysis.py
Facebook Recruiting IV: Human or Robot?
author: Yusuke Sakamoto

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

min_interval = 52631578.95


# plot time-lapse bids data by bots

bids_bots = pd.read_csv('data/bids_bots.csv')
info_bots = pd.read_csv('data/info_bots.csv', index_col=0)
bidder_ids = info_bots.index

bids_time_bots = []
fig = plt.figure()
for i in range(len(info_bots)):

    bids = bids_bots[bids_bots['bidder_id']==bidder_ids[i]]
    bids_time_bots.append(bids['time'])

    plt.plot(bids_time_bots[i], np.ones(len(bids_time_bots[i]))*i, '.')

plt.title('bots')
plt.xlabel('time')
plt.ylabel('bot#')
plt.xlim([9.62e15, 9.78e15])
plt.ylim([-1, len(info_bots)])
fig.savefig('figures/bids_time_bots.png')

####
# plot time-lapse bids data by humans


bids_humans = pd.read_csv('data/bids_humans.csv')
info_humans = pd.read_csv('data/info_humans.csv', index_col=0)
bidder_ids = info_humans.index
num_humans = len(info_humans)

num_itr = int(num_humans/100)+1

for k in range(num_itr):
    fig = plt.figure(k)

    print k
    bids_time_humans = []
    k_end = min(k*100+100, num_humans)
    
    for i in range(k*100, k_end):

        bids = bids_humans[bids_humans['bidder_id']==bidder_ids[i]]
        bids_time_humans.append(bids['time'])

        plt.plot(bids_time_humans[i-k*100],
                 np.ones(len(bids_time_humans[i-k*100]))*(i-k*100), '.')

    plt.title('humans #: %d ~ %d' %(k*100, k_end))
    plt.xlabel('time')
    plt.ylabel('human#')
    plt.xlim([9.62e15, 9.78e15])
    plt.ylim([-1, 101])
    fig.savefig('figures/bids_time_humans_%02d.png' %k)
    fig.show()
    plt.close(fig)
