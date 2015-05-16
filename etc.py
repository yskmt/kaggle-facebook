import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

bids_test = pd.read_csv('data/bids_test.csv')


def get_max_auc(bids):

    bidder_id = bids['bidder_id'].unique()
    num_bidders = len(bidder_id)

    # get the maximum number of auctions participated by one bidder
    auclen = []
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)

        bbbidder = bids[bids['bidder_id'] == bidder_id[i]]
        nbfea = []
        # count number of bids for each auction
        for auc in bbbidder['auction'].unique():
            nbfea.append(len(bbbidder[bbbidder['auction'] == auc]))

        auclen.append(len(nbfea))
    print max(auclen)

    return num_bidders, max(auclen)

# num_bidders, maxa = get_max_auc(bids_test)


def gather_info(num_bidders, max_auc, max_auc_count, bids, class_id):
    """
    Gather the useful infromation from bids data.
    """

    # ANALYSIS
    tmp_auc = np.zeros((num_bidders, max_auc), dtype=int)
    tmp = np.zeros((num_bidders, 7), dtype=int)
    tmp_mch = np.zeros(num_bidders, dtype=object)

    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # bids by this bidder
        bbbidder = bids[bids['bidder_id'] == class_id[i]]

        # number of bids by this bidder
        num_bbbidder = len(bbbidder)
        # number of auction by this bidder
        num_abbidder = len(bbbidder['auction'].unique())
        # number of merchandises by this bidder
        num_mbbidder = len(bbbidder['merchandise'].unique())
        # number of devices used by this bidder
        num_dbbidder = len(bbbidder['device'].unique())
        # number of countries by this bidder
        num_cbbidder = len(bbbidder['country'].unique())
        # number of ips by this bidder
        num_ibbidder = len(bbbidder['ip'].unique())
        # number of urls by this bidder
        num_ubbidder = len(bbbidder['url'].unique())

        # count number of bids for each auction
        nbfea = []
        for auc in bbbidder['auction'].unique():
            nbfea.append(len(bbbidder[bbbidder['auction'] == auc]))

        tmp_auc[i, :len(nbfea)] = sorted(nbfea, reverse=True)
        # NOTE: each bidder only has ONE unique merchandise, check
        # num_merchandise attribute
        tmp_mch[i] = bbbidder['merchandise'].unique()[0]

        tmp[i, 0] = num_bbbidder
        tmp[i, 1] = num_abbidder
        tmp[i, 2] = num_mbbidder
        tmp[i, 3] = num_dbbidder
        tmp[i, 4] = num_cbbidder
        tmp[i, 5] = num_ibbidder
        tmp[i, 6] = num_ubbidder

    bidders_mch = pd.get_dummies(pd.DataFrame(tmp_mch, index=class_id,
                                              columns=['merchandise']))
    bidders_info = pd.DataFrame(tmp, index=class_id,
                                columns=list(['num_bids',
                                              'num_aucs',
                                              'num_merchandise',
                                              'num_devices',
                                              'num_countries',
                                              'num_ips',
                                              'num_urls']))
    bidders_bids_by_aucs = pd.DataFrame(
        tmp_auc, index=class_id,
        columns=map(lambda x: 'num_bids_by_auc_' + str(x), range(max_auc)))

    bidders_info = pd.concat([bidders_info, bidders_mch,
                              bidders_bids_by_aucs.iloc[:, :max_auc_count]],
                             axis=1)

    return bidders_info, bidders_bids_by_aucs


def predict_usample(num_human, num_bots, human_info, bots_info, test_info,
                    holdout=0.0, multiplicity=5, plot_roc=False):
    '''
    multiplicity: ratio of human to bots in training set
    '''

    # under-sample the human data
    num_human_ext = min(num_bots * multiplicity, num_human)
    index_shuffle = range(num_human)
    np.random.shuffle(index_shuffle)

    if holdout > 0.0:
        num_human_train = int(num_human_ext * (1 - holdout))
        num_human_valid = num_human_ext - num_human_train

        human_train = human_info.iloc[index_shuffle[:num_human_train]]
        human_valid = human_info.iloc[
            index_shuffle[num_human_train:num_human_ext]]

        num_bots_train = int(num_bots * (1 - holdout))
        num_bots_valid = num_bots - num_bots_train

        bots_train = bots_info.iloc[:num_bots_train]
        bots_valid = bots_info.iloc[num_bots_train:]

        train_info = pd.concat([human_train, bots_train], axis=0).sort(axis=1)
        valid_info = pd.concat([human_valid, bots_valid], axis=0).sort(axis=1)
    else:
        num_human_train = num_human_ext
        num_bots_train = num_bots
        train_info = pd.concat(
            [human_info.iloc[index_shuffle[:num_human_ext]], bots_info],
            axis=0).sort(axis=1)

    X_train = train_info.values[:, :].astype(float)
    y = np.concatenate(
        [np.zeros(num_human_train), np.ones(num_bots_train)], axis=0)

    # shuffle!
    index_shuffle = range(len(y))
    np.random.shuffle(index_shuffle)
    X_train = X_train[index_shuffle]
    y = y[index_shuffle]

    X_test = test_info.values[:, :]

    # Predict!
    # print "fitting the model"
    clf = RandomForestClassifier(n_estimators=100, n_jobs=2,
                                 random_state=1234, verbose=0,
                                 max_features='auto')
    # clf = GradientBoostingClassifier()
    # clf = SGDClassifier(loss="log", verbose=1, random_state=1234, n_iter=5000)
    # clf = LogisticRegression()
    clf.fit(X_train, y)

    # prediction on the validation set
    if holdout > 0.0:
        X_valid = valid_info.values[:, :].astype(float)
        y_valid = np.concatenate(
            [np.zeros(num_human_valid), np.ones(num_bots_valid)], axis=0)

        insh = range(len(y_valid))
        np.random.shuffle(insh)
        X_valid = X_valid[insh]
        y_valid = y_valid[insh]

        valid_proba = clf.predict_proba(X_valid)
        valid_pred = clf.predict(X_valid)

        fpr, tpr, thresholds = roc_curve(y_valid, valid_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        print "Area under the ROC curve : %f" % roc_auc
        
        if plot_roc:
            # Plot ROC curve
            plt.clf()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
    else:
        auc_valid = 0.0
        y_valid = 0.0
        roc_auc = 0.0
        
    # prediction on test set
    y_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    # measuring prediction peformance agianst train set
    train_proba = clf.predict_proba(X_train)
    train_pred = clf.predict(X_train)

    return y_proba, y_pred, train_proba, train_pred, roc_auc
