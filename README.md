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


## Second strategy

Useful information to be extracted:

1. Total number of bids
2. Total number of auctions participated
3. Number of bids in each auction: should be sorted by descending order
   and 0 for no bid.
4. Numbed of __ by a bidder
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

holdout=0.1
[[  1.      5.      9.     13.     17.   ]
 [  0.933   0.946   0.941   0.949   0.944]
 [  0.062   0.021   0.017   0.012   0.007]
 [  0.221   0.068   0.049   0.037   0.035]
 [  0.053   0.026   0.025   0.014   0.018]]

holdout=0.2
[[  1.      5.      9.     13.     17.   ]
 [  0.914   0.917   0.913   0.91    0.907]
 [  0.035   0.019   0.018   0.014   0.014]
 [  0.195   0.069   0.036   0.025   0.024]
 [  0.048   0.029   0.025   0.012   0.017]]

holdout=0.0 --> multiplicity=8?



1. max_auc_count = 1500
bots proba for test set:  0.09493412527
[[ 1.   ]
 [ 0.91 ]
 [ 0.045]
 [ 0.095]
 [ 0.066]
 [ 0.888]
 [ 0.058]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.051403887689

[[ 1.   ]
 [ 0.915]
 [ 0.042]
 [ 0.097]
 [ 0.059]
 [ 0.899]
 [ 0.053]
 [ 0.845]
 [ 0.054]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.0520518358531

[[ 10.   ]
 [  0.915]
 [  0.014]
 [  0.003]
 [  0.003]
 [  0.37 ]
 [  0.078]
 [  0.922]
 [  0.011]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.000215982721382
multiplicity       1.000
roc_auc: mean      0.911
roc_auc: std       0.049
bots_rate: mean    0.072
bote_rate: std     0.057
specificity: mean  0.889
specificity: std   0.077
accuracy: mean     0.839
accuracy: std      0.057
bots proba for train set: 0.0519153225806
bots proba for test set:  0.0207343412527


2. max_auc_count = 150
CV result:
[[ 1.   ]
 [ 0.913]
 [ 0.04 ]
 [ 0.078]
 [ 0.045]
 [ 0.906]
 [ 0.049]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.0427645788337

                       0
multiplicity       1.000
roc_auc: mean      0.914
roc_auc: std       0.045
bots_rate: mean    0.075
bote_rate: std     0.051
specificity: mean  0.897
specificity: std   0.076
accuracy: mean     0.847
accuracy: std      0.055
bots proba for train set: 0.0519153225806
bots proba for test set:  0.0313174946004


3. max_auc_count = 0
CV result:
[[ 1.   ]
 [ 0.907]
 [ 0.044]
 [ 0.255]
 [ 0.028]
 [ 0.899]
 [ 0.048]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.260691144708

4. max_auc_count = 0
dropped all the num_* columns except num_bids

CV result:
[[ 1.   ]
 [ 0.875]
 [ 0.049]
 [ 0.242]
 [ 0.026]
 [ 0.827]
 [ 0.065]
 [ 0.812]
 [ 0.054]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.241252699784

5. max_auc_count = 0
dropped all the num_* columns except num_bids
dropped all the merchandise_* labels

CV result:
[[ 1.   ]
 [ 0.897]
 [ 0.047]
 [ 0.23 ]
 [ 0.023]
 [ 0.889]
 [ 0.05 ]
 [ 0.844]
 [ 0.051]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.244060475162

6 - just use num_bids

CV result:
[[ 1.   ]
 [ 0.857]
 [ 0.066]
 [ 0.231]
 [ 0.017]
 [ 0.815]
 [ 0.084]
 [ 0.805]
 [ 0.063]]
bots proba for train set: 0.0519153225806
bots proba for test set:  0.27494600432
