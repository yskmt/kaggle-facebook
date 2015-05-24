"""
analysis.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import pandas as pd

from analysis_funcs import (gather_info, gather_info_by_periods,
                            gather_country_info,
                            gather_auc_bids_info, gather_device_info,
                            gather_count_info)

# set(bids_test['merchandise'].unique())\
#     .union(set(bids_humans['merchandise'].unique()))\
#     .union(set(bids_bots['merchandise']))
merchandise_keys = ['auto parts',
                    'books and music',
                    'clothing',
                    'computers',
                    'furniture',
                    'home goods',
                    'jewelry',
                    'mobile',
                    'office equipment',
                    'sporting goods']

bids_humans = pd.read_csv('data/bids_humans.csv', index_col=0)
bids_bots = pd.read_csv('data/bids_bots.csv', index_col=0)
bids_test = pd.read_csv('data/bids_test.csv', index_col=0)

############################################################################
# Gathering basic counts information
############################################################################

# print "Analyzing huaman data..."
# info_humans = gather_info(bids_humans)
# info_humans.to_csv('data/info_humans.csv')

# print "Analyzing huaman data..."
# info_bots = gather_info(bids_bots)
# info_bots.to_csv('data/info_bots.csv')

# print "Analyzing huaman data..."
# info_test = gather_info(bids_test)
# info_test.to_csv('data/info_test.csv')

############################################################################
# Gathering country counts information
############################################################################

# print "Analyzing country huaman data..."
# cinfo_humans = gather_country_info(bids_humans)
# cinfo_humans.to_csv('data/country_info_humans.csv')

# print "Analyzing country huaman data..."
# cinfo_bots = gather_country_info(bids_bots)
# cinfo_bots.to_csv('data/country_info_bots.csv')

# print "Analyzing country huaman data..."
# cinfo_test = gather_country_info(bids_test)
# cinfo_test.to_csv('data/country_info_test.csv')

# cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
# cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
# cinfo_test = pd.read_csv('data/country_info_test.csv', index_col=0)

############################################################################
# Gathering device counts information
############################################################################

# print "Analyzing device huaman data..."
# dinfo_humans = gather_device_info(bids_humans)
# dinfo_humans.to_csv('data/device_info_humans.csv')

# print "Analyzing device huaman data..."
# dinfo_bots = gather_device_info(bids_bots)
# dinfo_bots.to_csv('data/device_info_bots.csv')

# print "Analyzing device huaman data..."
# dinfo_test = gather_device_info(bids_test)
# dinfo_test.to_csv('data/device_info_test.csv')

# dinfo_humans = pd.read_csv('data/device_info_humans.csv', index_col=0)
# dinfo_bots = pd.read_csv('data/device_info_bots.csv', index_col=0)
# dinfo_test = pd.read_csv('data/device_info_test.csv', index_col=0)


############################################################################
# Gathering url counts information
############################################################################

# print "Analyzing url huaman data..."
# dinfo_humans = gather_count_info(bids_humans, 'url')
# dinfo_humans.to_csv('data/url_info_humans.csv')

# print "Analyzing url huaman data..."
# dinfo_bots = gather_count_info(bids_bots, 'url')
# dinfo_bots.to_csv('data/url_info_bots.csv')

# print "Analyzing url huaman data..."
# dinfo_test = gather_count_info(bids_test, 'url')
# dinfo_test.to_csv('data/url_info_test.csv')

# dinfo_humans = pd.read_csv('data/url_info_humans.csv', index_col=0)
# dinfo_bots = pd.read_csv('data/url_info_bots.csv', index_col=0)
# dinfo_test = pd.read_csv('data/url_info_test.csv', index_col=0)


############################################################################
# Gathering bids-by-aucs counts information
############################################################################

# print "Analyzing bids-by-aucs humans data..."
# bbainfo_humans = gather_auc_bids_info(bids_humans)
# bbainfo_humans.to_csv('data/bba_info_humans.csv', index_label='bidder_id')

# print "Analyzing bids-by-aucs bots data..."
# bbainfo_bots = gather_auc_bids_info(bids_bots)
# bbainfo_bots.to_csv('data/bba_info_bots.csv', index_label='bidder_id')

# print "Analyzing bids-by-aucs test data..."
# bbainfo_test = gather_auc_bids_info(bids_test)
# bbainfo_test.to_csv('data/bba_info_test.csv', index_label='bidder_id')

# cinfo_humans = pd.read_csv('data/country_info_humans.csv', index_col=0)
# cinfo_bots = pd.read_csv('data/country_info_bots.csv', index_col=0)
# cinfo_test = pd.read_csv('data/country_info_test.csv', index_col=0)


############################################################################
# Gathering basic counts information by each period
############################################################################

print "Analyzing huaman basic count data per each period..."
info_humans_bp = gather_info_by_periods(bids_humans)
info_humans_bp.to_csv('data/info_humans_bp.csv')

print "Analyzing huaman basic count data per each period..."
info_bots_bp = gather_info_by_periods(bids_bots)
info_bots_bp.to_csv('data/info_bots_bp.csv')

print "Analyzing huaman basic count data per each period..."
info_test_bp = gather_info_by_periods(bids_test)
info_test_bp.to_csv('data/info_test_bp.csv')


############################################################################
# statistical significance of the difference of distributions between
# humans and bots sets
############################################################################
# from scipy.stats import ttest_ind

# ctkeys = cinfo_humans.keys().union(cinfo_bots.keys())
# keys_sig = []
# keys_na = []

# for key in ctkeys:
#     if (key in cinfo_bots.keys()) and (key in cinfo_humans.keys()):

#         t, prob = ttest_ind(
#             cinfo_humans[key].values, cinfo_bots[key].values, equal_var=False)
#         if prob < 0.05:
#             print key, prob
#             keys_sig.append(key)

#     else:
#         print key
#         keys_na.append(key)

# keys_sig
# ['ar', 'au', 'bd', 'dj', 'ga', 'gq', 'id', 'mc', 'ml', 'mr', 'mz', 'nl', 'th']

# keys_na
# ['an', 'aw', 'bi', 'cf', 'er', 'gi', 'gn', 'gp', 'mh', 'nc', 'sb', 'tc', 'vi', 'ws']
