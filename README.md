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
* Bagging of the extratree is not the way to go...

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

