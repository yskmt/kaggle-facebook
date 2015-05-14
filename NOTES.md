# Facebook Recruiting IV: Human or Robot?


## Strategies

1. Classify whether **each bid** is done by a bot or human. Gather all the
   bids by the same bidder_id and average the probability (or
   classification).
2. Gather each bidder's (with the same bidder_id) information such as
   number of bids, number of auction, time, etc, and classify whether
   **each bidder** is a bot or not.

## Observations

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
* Number of merchandies by the same bidder is always **1**.

## Second strategy

Useful information to be extracted:

1. Total number of bids
2. Total number of auctions participated
3. Number of bids in each auction: should be sorted by descending order
   and 0 for no bid.
4. Bid frequency extracted from time
   * Bid freqnency extracted from time for each **device**
5. Numbef of auctions participated for each merchandise type 
   * Number of merchandise by each bidder is always **1**.
6. Number of devices used
7. Countries, urls, devices: check the correlation between each label and bot/human
   classification and use the best ~10 countries?
8. Auction winners(?)
   * Last bidder in a given auction.
9. Price of the item bidders bid
   * Total number of bids in each auction == price of the item
10. Anomaly behaviors(?)
    * Repeated bids by one person

## Problesm

1. Generating dummy vaiables take up a large amount of memory.
   *  Brute-force dummy labeling will create 1,425,220 labels
#  

## Ideas

* time can be categorize into smaller groups?
* cluster the dataset using some unsupervised learning technique?

