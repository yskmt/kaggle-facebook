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







# features

most important 400/7766 featuers by ET
['58', '8', 'phone46', '7', '3', '9', 'bba_4', '6', '2', 'bba_5', '4', '10', 'bba_3', 'au', 'bba_6', '11', '1', 'bba_2', 'phone143', '19', '12', 'num_bids', '5', 'phone1030', 'phone195', 'phone55', '64', '18', '13', 'sg', '22', '46', 'bba_8', 'phone144', '52', '62', 'phone63', '61', 'bba_1', 'bba_7', '50', '20', '0', '27', '60', '54', 'bba_9', '21', '30', 'bba_10', '93', '17', '32', '35', '33', '34', '67', '70', '47', '36', 'num_ips', '76', 'num_devices', '16', '26', '57', '25', '24', '72', '29', '37', 'bba_11', '23', '55', '81', 'phone184', '15', '53', 'bba_12', '92', 'phone739', '43', '80', 'bba_14', '86', 'phone110', 'num_aucs', 'za', 'num_countries', '82', '66', '56', '31', '75', 'bba_13', 'phone792', '68', '78', 'bba_16', '89', '97', 'bba_15', '38', '28', '44', '39', '14', '87', '79', '71', '48', '85', 'phone33', '90', '59', 'phone2133', '49', '74', '77', '63', 'phone1026', 'ph', '91', '45', 'phone5', '73', 'num_urls', 'phone25', '84', 'mobile', '69', '88', 'bba_17', '83', 'phone28', 'phone219', 'th', 'phone2099', '40', 'phone469', '98', '94', 'fr', '99', 'de', 'phone150', 'uk', 'my', 'phone4', 'id', 'bba_20', 'phone607', 'bba_18', 'phone640', 'phone6', 'phone124', 'phone892', 'bba_21', '95', 'phone1166', 'phone728', 'phone2330', '42', 'computers', 'bba_19', 'phone136', '51', 'bba_22', '41', '96', 'it', 'bba_23', '65', 'phone2287', 'bba_24', 'phone938', 'phone22', 'sporting goods', 'in', 'phone1013', 'phone899', 'phone21', 'jewelry', 'bba_25', 'phone168', 'phone205', 'phone280', 'phone212', 'bba_26', 'phone178', 'phone4479', 'phone5479', 'bba_29', 'phone6273', 'phone5936', 'cn', 'bba_27', 'kw', 'phone996', 'bba_30', 'bba_28', 'phone576', 'phone65', 'phone706', 'phone1898', 'home goods', 'phone2341', 'ca', 'phone152', 'phone11', 'ar', 'phone2229', 'bba_31', 'ch', 'phone106', 'phone2955', 'phone764', 'phone4171', 'phone3', 'phone1270', 'phone185', 'us', 'phone546', 'phone361', 'phone45', 'phone243', 'no', 'phone57', 'phone424', 'phone656', 'phone6463', 'phone1814', 'nl', 'office equipment', 'phone4721', 'bba_33', 'phone2131', 'phone125', 'jp', 'phone1683', 'phone2727', 'phone431', 'phone101', 'bba_32', 'bba_36', 'bba_34', 'phone35', 'phone69', 'phone5620', 'bba_37', 'bba_35', 'ua', 'phone2123', 'phone339', 'phone47', 'phone2174', 'phone2', 'phone419', 'phone887', 'phone790', 'phone16', 'phone58', 'bba_39', 'phone720', 'phone534', 'phone5886', 'phone3952', 'bba_41', 'bba_38', 'bba_43', 'bba_40', 'phone762', 'ke', 'phone5658', 'phone3584', 'phone36', 'mx', 'bba_44', 'phone312', 'phone652', 'bba_42', 'phone226', 'phone1046', 'phone1525', 'phone1120', 'phone1306', 'phone26', 'phone387', 'phone588', 'bba_45', 'phone5898', 'phone159', 'phone2109', 'br', 'phone199', 'ng', 'books and music', 'phone5180', 'bn', 'bba_49', 'qa', 'bba_48', 'bba_47', 'phone1104', 'phone2612', 'phone80', 'phone4230', 'phone281', 'phone598', 'tr', 'phone59', 'phone689', 'phone5723', 'bba_46', 'phone248', 'phone41', 'phone20', 'bba_50', 'phone153', 'phone4512', 'phone3317', 'phone5676', 'phone142', 'phone2877', 'phone448', 'phone651', 'phone5115', 'tw', 'phone77', 'phone8', 'phone76', 'phone72', 'phone162', 'phone1710', 'phone3287', 'phone3359', 'phone2115', 'phone315', 'phone91', 'phone90', 'ru', 'phone1519', 'phone2698', 'phone38', 'ma', 'phone348', 'bba_55', 'phone1917', 'phone516', 'phone2295', 'phone252', 'bba_52', 'phone220', 'phone13', 'phone1697', 'es', 'phone316', 'phone1043', 'phone98', 'bba_54', 'phone15', 'phone1062', 'phone364', 'phone176', 'phone3433', 'phone224', 'sa', 'phone1946', 'bba_57', 'phone1', 'phone179', 'bba_61', 'bba_51', 'bba_59', 'phone97', 'phone93', 'phone1197', 'phone94', 'phone2001', 'phone1482', 'phone83', 'phone871', 'bba_53', 'phone321', 'phone4496', 'phone872', 'fi', 'bba_58', 'phone297', 'phone785', 'phone169', 'phone164', 'phone99', 'hk', 'phone2571', 'phone5505', 'eg', 'ie', 'phone1320', 'bba_56']
