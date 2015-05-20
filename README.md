# Facebook Recruiting IV: Human or Robot?

* pandas: http://pandas.pydata.org/pandas-docs/stable/10min.html

## Strategies

1. Classify whether **each bid** is done by a bot or human. Gather all the
   bids by the same bidder_id and average the probability (or
   classification).
2. Gather each bidder's (with the same bidder_id) information such as
   number of bids, number of auction, time, etc, and classify whether
   **each bidder** is a bot or not.

## Observations

* UNBALANCED SAMPLES!!! The number of bots are far fewer than number
  of human. Algorithm needs to accommodate this unbalanced-ness.
    * http://stats.stackexchange.com/questions/17225/when-over-under-sampling-unbalanced-classes-does-maximizing-accuracy-differ-fro
    * http://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/
* Number of bids done by bidders in training data: 3071224
* Number of unique __ done by bidders in the training dataset
    * auctions: 12740
    * merchandises: 10
    * devices 5729
    * time: 742669
    * country: 199
    * url: 663873
* Bots only participate auction with those categories:
    * mobile, jewelry, office equipment, computers books and
    music, sporting goods, home goods
* Each auction can have multiple merchandises
    * "Merchandise" category is merely a search term that each bid is
    referred.
* There are bidders (both in train.csv, test.csv) that do not have any
  bids in bids.csv.
* Number of merchandises by the same bidder is always **1**.
* **Auction infromation need to be extracted**
    * Number of bidders who participated in an auction
    * End price of the merchandise
    * Who won (the last one to bid) the auction
    * Who enter the existing auction
    * Probably just those that bidders from train/test sets are
      involved.
* Time-lapse data
    * Frequency of the bids
    * Length of the bid streaks
      * Calculate the difference in time between each bid and if it is
        smllaer then a certin thershold, then it is consideres wtihin
        a stretk.
* Feature selection
    * Student t-test
    * Chi2 test
    * Get the correlation among features and remove the one with high
      correlation.

## Outliers

* bots with only one bid:
7fab82fa5eaea6a44eb743bc4bf356b3tarle
f35082c6d72f1f1be3dd23f949db1f577t6wd
bd0071b98d9479130e5c053a244fe6f1muj8h
91c749114e26abdb9a4536169f9b4580huern
74a35c4376559c911fdb5e9cfb78c5e4btqew


## Second strategy

Useful information to be extracted:

1. Total number of bids
2. Total number of auctions participated
3. Number of bids in each auction: should be sorted by descending order
   and 0 for no bid.
4. Number of __ by a bidder
    * auctions participated for each merchandise type 
    * devices
    * countries
    * ips
    * urls
    * merchandises (always 1)
5. Bid frequency extracted from time
    * Bid frequency extracted from time for each **device**
6. Countries, urls, devices: check the correlation between each label and bot/human
   classification and use the best ~10 countries?
7. Auction winners(?)
    * Last bidder in a given auction.
8. Price of the item bidders bid
    * Total number of bids in each auction == price of the item
9. Anomaly behaviors(?)
    * Repeated bids by one person

## Problem

1. Generating dummy variables take up a large amount of memory.
    * Brute-force dummy labeling will create 1,425,220 labels


## Ideas

* time can be categorize into smaller groups?
* cluster the dataset using some unsupervised learning technique?


## CV results

Feature ranking:
0. feature 0: au = (0.134989)
1. feature 8: num_devices = (0.111759)
2. feature 6: id = (0.105308)
3. feature 5: bba_5 = (0.102224)
4. feature 4: bba_4 = (0.099977)
5. feature 9: th = (0.094807)
6. feature 2: bba_2 = (0.089691)
7. feature 7: num_bids = (0.087557)
8. feature 1: bba_1 = (0.087204)
9. feature 3: bba_3 = (0.086484)


n_st=500
0.91540721992 0.0203362966484
0.954939205096 0.00133639367936

n_est=4000
0.915271602114 0.0198226221489
0.953627599105 0.00242965319325

n_est=500
outlier removed:
n_features=10
0.929231850808 0.0156712434759
0.958162777538 0.001760586993

n_est=1000
outlier removed:
n_features=22
0.940344774491 0.0194237716926
0.959678707813 0.000498810944577

n_est = 1000
['bba_3', 'bba_2', 'num_bids', 'bba_1', 'num_ips', 'phone46', 'au',
'phone143', 'phone28', 'th', 'phone13', 'phone17', 'phone62', 'phone290',
'phone157', 'phone479', 'phone237', 'phone346', 'phone248', 'phone119', 
'phone56', 'phone122']

0.927225232133 0.0208572435834
0.959880471005 0.00168220270904


n_est = 1000
chose 10*4 most important features from each category by ET
['num_bids', 'au', 'num_aucs', 'bba_15', 'num_ips', 'bba_14', 'num_devices',
 'num_countries', 'bba_17', 'num_urls', 'phone150', 'bba_18', 'bba_19',
 'phone55', 'phone33', 'bba_30', 'phone739', 'phone1030', 'my', 'bba_31',
 'bba_35', 'phone136', 'ca', 'bba_32', 'bba_33', 'cn', 'phone58', 'phone640',
 'za', 'in', 'phone15', 'phone996', 'ar', 'ru', 'ch', 'ec']
0.945600450453 0.0169479757094
0.960789561914 0.000756052446392


16. 
n_est = 2000
0.948751144697 0.015679560222
0.964327489414 0.00113223205984
['au', 'phone46', 'phone55', 'za', 'phone739', 'bba_4', 'num_devices', 'bba_5', 'my', 'de', 'bba_2', 'th', 'bba_6', 'bba_9', 'bba_3', 'num_bids', 'ca', 'bba_7', 'bba_8', 'bba_1', 'phone640', 'us', 'cn', 'num_urls', 'phone996', 'jp', 'phone136', 'num_countries', 'in', 'num_aucs', 'phone150', 'ch', 'num_ips', 'ar', 'bba_15', 'bba_14', 'phone33', 'bba_17', 'phone1030', 'phone58']


17
n_est = 2000
0.95383276018 0.0157806346745
0.962807988053 0.00133983098017
['au', 'phone46', 'phone143', 'phone739', 'phone55', 'za', 'sg', 'phone728', 'bba_4', 'bba_5', 'th', 'uk', 'num_devices', 'jp', 'bba_2', 'ca', 'ph', 'de', 'phone2287', 'it', 'my', 'bba_6', 'phone996', 'id', 'phone136', 'bba_3', 'phone21', 'phone110', 'bba_8', 'us', 'num_bids', 'bba_7', 'bba_1', 'phone28', 'bba_10', 'bba_9', 'phone63', 'phone640', 'cn', 'num_countries', 'bba_11', 'num_aucs', 'ch', 'fr', 'phone1026', 'num_urls', 'phone195', 'in', 'ar', 'phone150', 'phone469', 'phone219', 'phone144', 'bba_12', 'phone6', 'phone168', 'num_ips', 'phone25', 'phone2330', 'phone58', 'bn', 'bba_15', 'phone65', 'bba_14', 'phone33', 'phone1030', 'bba_13', 'bba_16', 'nl', 'phone22', 'ua', 'br', 'bba_17', 'phone3359', 'ru', 'no', 'phone90', 'bba_19', 'phone224', 'bba_21', 'bba_22', 'phone205', 'bh', 'bba_20', 'bba_18', 'bba_26', 'qa', 'bba_29', 'bba_24', 'bba_28', 'bba_27', 'bba_23', 'bba_25', 'bba_30', 'bba_32', 'bba_33', 'computers', 'lt', 'sa', 'bba_31', 'bba_35', 'mobile', 'phone15', 'sporting goods', 'lv', 'tw', 'jewelry', 'home goods', 'lu', 'kr', 'phone239', 'ec', 'office equipment', 'books and music', 'bf', 'auto parts', 'furniture', 'clothing', 'num_merchs']

