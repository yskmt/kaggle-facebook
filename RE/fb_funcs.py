import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation

import xgboost as xgb

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

# countries that can be significant: derived in analysis.py
keys_sig = ['ar', 'au', 'bd', 'dj', 'ga', 'gq', 'id', 'mc', 'ml',
            'mr', 'mz', 'nl', 'th']
keys_na = ['an', 'aw', 'bi', 'cf', 'er', 'gi', 'gn', 'gp', 'mh', 'nc',
           'sb', 'tc', 'vi', 'ws']


def predict_usample(num_humans, num_humans_use, num_bots_use,
                    info_humans, info_bots, p_valid=0.2, plot_roc=False):
    """
    prediction by undersampling

    p_valid: validation set fraction
    """

    index_h = np.random.choice(num_humans, num_humans_use, replace=False)
    info_humans_use = info_humans.iloc[index_h, :]

    # combine humans and bots data to create given data
    info_given = pd.concat([info_humans_use, info_bots], axis=0)
    labels_train = np.hstack((np.zeros(num_humans_use), np.ones(num_bots_use)))

    # split into training and validation sets
    num_given = len(info_given)
    num_valid = int(num_given * p_valid)
    num_train = num_given - num_valid

    index_vt = np.random.choice(num_given, num_given, replace=False)

    info_valid = info_given.iloc[index_vt[:num_valid], :]
    info_train = info_given.iloc[index_vt[num_valid:], :]

    X_valid = info_valid.as_matrix()
    X_train = info_train.as_matrix()

    y_valid = labels_train[index_vt[:num_valid]]
    y_train = labels_train[index_vt[num_valid:]]

    # randomforest!
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, y_train)
    y_valid_proba = clf.predict_proba(X_valid)
    y_valid_pred = clf.predict(X_valid)

    fpr, tpr, thresholds = roc_curve(y_valid, y_valid_proba[:, 1])
    roc_auc = auc(fpr, tpr)

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

    clf_score = clf.score(X_valid, y_valid)

    # true positive rate at 0.5 threshold
    tpr_50 = (
        tpr[sum(thresholds > 0.5)] + tpr[sum(thresholds > 0.5) - 1]) / 2.0

    return clf, roc_auc, clf_score, tpr_50


def predict_cv(info_humans, info_bots, plot_roc=False,
               n_folds=5, n_estimators=1000):
    """
    prediction by undersampling

    p_valid: validation set fraction
    """

    num_humans = len(info_humans)
    num_bots = len(info_bots)

    # combine humans and bots data to create given data
    info_given = pd.concat([info_humans, info_bots], axis=0)
    labels_train = np.hstack((np.zeros(num_humans), np.ones(num_bots)))
    num_given = len(labels_train)

    # shuffle just in case
    index_sh = np.random.choice(num_given, num_given, replace=False)
    info_given = info_given.iloc[index_sh]
    labels_train = labels_train[index_sh]

    # get matrices forms
    X = info_given.sort(axis=1).as_matrix()
    y = labels_train

    # split for cv
    kf = cross_validation.StratifiedKFold(
        y, n_folds=n_folds, shuffle=True, random_state=None)

    # cv scores
    roc_auc = np.zeros(n_folds)
    clf_score = np.zeros(n_folds)
    tpr_50 = np.zeros(n_folds)

    n_cv = 0
    for train_index, test_index in kf:
        print "CV#: ", n_cv
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # clf = RandomForestClassifier(n_estimators=n_estimators,
        # class_weight=None, max_features=None)
        clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features=0.015)

        # clf = SGDClassifier(loss='log')
        # clf = DecisionTreeClassifier()

        clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)
        # y_test_pred = clf.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
        roc_auc[n_cv] = auc(fpr, tpr)

        if plot_roc:
            # Plot ROC curve
            # plt.clf()
            plt.plot(fpr, tpr, label='ROC curve # %d (area = %0.2f)' %
                     (n_cv, roc_auc[n_cv]))

            # plt.show()

        clf_score[n_cv] = clf.score(X_test, y_test)

        # true positive rate at 0.5 threshold
        tpr_50[n_cv] = (
            tpr[sum(thresholds > 0.5)] + tpr[sum(thresholds > 0.5) - 1]) / 2.0

        n_cv += 1

    if plot_roc:
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    return clf, roc_auc, clf_score, tpr_50


def fit_and_predict(info_humans, info_bots, info_test,
                    n_estimators=1000, p_use=None, model="ET",
                    plot_importance=False):

    num_humans = len(info_humans)
    num_bots = len(info_bots)
    # num_test = len(info_test)

    if p_use is not None:
        num_bots_use = int(num_bots)
        num_humans_use = int(num_humans * p_use)

        indx_bots = np.random.choice(num_bots, num_bots_use, replace=False)
        indx_humans = np.random.choice(
            num_humans, num_humans_use, replace=False)

        info_humans = info_humans.iloc[indx_humans]
        info_bots = info_bots.iloc[indx_bots]

        num_bots = num_bots_use
        num_humans = num_humans_use

    # combine humans and bots data to create given data
    info_given = pd.concat([info_humans, info_bots], axis=0)
    labels_train = np.hstack((np.zeros(num_humans), np.ones(num_bots)))
    num_given = len(labels_train)

    # shuffle just in case
    index_sh = np.random.choice(num_given, num_given, replace=False)
    info_given = info_given.iloc[index_sh]
    labels_train = labels_train[index_sh]

    # get matrices forms
    X_train = info_given.sort(axis=1).as_matrix()
    y_train = labels_train
    X_test = info_test.sort(axis=1).as_matrix()

    features = info_given.sort(axis=1).keys()

    if model == 'RF':
        # randomforest!
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     class_weight='auto', verbose=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)
        return y_pred[:, 1], y_train_pred[:, 1], 0

    elif model == 'ET':
        clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                   max_features=0.015,
                                   random_state=0)
        clf.fit(X_train, y_train)
        importances = clf.feature_importances_

        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(min(len(features), 40)):
            print("%d. feature %d: %s = (%f)"
                  % (f, indices[f], features[indices[f]], importances[indices[f]]))

        # print list((features[indices])[:22])
        print list((features[indices]))[:40]

        # Plot the feature importances of the forest
        if plot_importance:
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(len(features)), importances[indices],
                    color="r", yerr=std[indices], align="center")
            plt.xticks(range(len(features)), features[indices])
            plt.xlim([-1, len(features)])
            plt.show()

        y_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)

        return y_pred[:, 1], y_train_pred[:, 1], 0, list((features[indices]))
    elif model == "XGB":
        # xgboost!
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {"objective": "binary:logistic", "eta": 0.01, "max_depth": 10}
        num_rounds = n_estimators

        evallist = [(dtrain, 'train')]
        bst = xgb.train(params, dtrain, num_rounds, evallist)
        dtest = xgb.DMatrix(X_test)
        y_pred = bst.predict(dtest)
        y_train_pred = bst.predict(dtrain)

        return y_pred, y_train_pred, y_train, 0

    elif model == "XGB_CV":
        cv_result = xgb.cv(params, dtrain, num_rounds, nfold=cv,
                           metrics={'rmse', 'error', 'auc'}, seed=0)
        return 0, 0, cv_result

def append_merchandise(info, drop=True):
    """
    Append merchandises a dummy variable and drop if needed.
    """

    merchandise = info.merchandise
    merchandise = pd.get_dummies(merchandise)
    for key in merchandise_keys:
        if key not in merchandise.keys():
            print key, "added."
            merchandise[key] \
                = pd.Series(np.zeros(len(merchandise)),
                            index=merchandise.index)
    info = pd.concat([info, merchandise], axis=1)

    if drop == True:
        info.drop('merchandise', axis=1, inplace=True)

    return info


def append_bba(info, bbainfo, num_bba):
    """
    Append bids-by-auction data
    """

    return pd.concat([info, bbainfo.iloc[:, :num_bba]], axis=1)


def append_countries(info, cinfo, countries):
    """
    Append country counts

    info: bidder info
    cinfo: bidder country info
    countries: countries to be appended
    """

    countries_appended = []
    for key in countries:
        # for key in list(cinfo.keys()):
        if key in list(cinfo.keys()):
            countries_appended.append(cinfo[key])
        else:
            # just create zero-column
            countries_appended.append(
                pd.DataFrame(np.zeros(len(cinfo)), index=cinfo.index, columns=[key]))

    countries_appended.append(info)

    info = pd.concat(countries_appended, axis=1)

    return info


def append_device(info, dinfo, devices):
    """
    Append device counts

    info: bidder info
    dinfo: bidder country info
    countries: countries to be appended
    """

    devices_appended = []
    for key in devices:
        # for key in list(dinfo.keys()):
        if key in list(dinfo.keys()):
            devices_appended.append(dinfo[key])
        else:
            # just create zero-column
            devices_appended.append(
                pd.DataFrame(np.zeros(len(dinfo)), index=dinfo.index, columns=[key]))

    devices_appended.append(info)

    info = pd.concat(devices_appended, axis=1)

    return info


def append_bids_intervals(info, biinfo, bids_intervals):
    """
    Append bids interval data

    info: bidder info
    biinfo: bidder country info
    bids_intervals: bids_intervals to be appended
    """

    bids_intervals_appended = []
    for key in bids_intervals:
        if key in list(biinfo.keys()):
            bids_intervals_appended.append(biinfo[key])
        else:
            # just create zero-column
            bids_intervals_appended.append(
                pd.DataFrame(np.zeros(len(biinfo)),
                             index=biinfo.index, columns=[key]))

    bids_intervals_appended.append(info)

    info = pd.concat(bids_intervals_appended, axis=1)

    return info



def gather_info(bids_data):
    """
    Gather the useful infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    # tmp = np.zeros((num_bidders, 7), dtype=int)
    # tmp_mch = np.zeros(num_bidders, dtype=object)

    bidders_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of bids by this bidder
        num_bids = len(bids)
        # number of auction by this bidder
        num_aucs = len(bids['auction'].unique())
        # number of merchandises by this bidder
        num_merchs = len(bids['merchandise'].unique())
        # number of devices used by this bidder
        num_devices = len(bids['device'].unique())
        # number of countries by this bidder
        num_countries = len(bids['country'].unique())
        # number of ips by this bidder
        num_ips = len(bids['ip'].unique())
        # number of urls by this bidder
        num_urls = len(bids['url'].unique())

        # primary merchandise. NOTE: every bidder only has one merchandise
        merch = bids['merchandise'].unique()[0]

        tmp_info = np.array([num_bids, num_aucs, num_merchs,
                             num_devices, num_countries, num_ips, num_urls,
                             merch])

        tmp_info = tmp_info.reshape(1, len(tmp_info))

        bidders_info.append(pd.DataFrame(
            tmp_info,  index=[bidder_ids[i]],
            columns=['num_bids', 'num_aucs', 'num_merchs', 'num_devices',
                     'num_countries', 'num_ips', 'num_urls', 'merchandise']
        ))

    bidders_info = pd.concat(bidders_info, axis=0)
    bidders_info.index.name = 'bidder_id'

    return bidders_info


def gather_auc_bids_info(bids_data):
    """
    Gather the number of bids for each auction info.
    The number of bids are sorted in descending order.
    """

    # ANALYSIS
    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    # for each bidder
    bidders_aucbids_info = []
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # count number of bids for each auction by this bidder
        num_bids_auction = []
        for auction in bids['auction'].unique():
            num_bids_auction.append(len(bids[bids['auction'] == auction]))

        bidders_aucbids_info.append(sorted(num_bids_auction, reverse=True))

    bbainfo_bots = pd.DataFrame(bidders_aucbids_info, index=bidder_ids)
    bbainfo_bots.fillna(0, inplace=True)

    # change column label to reflect the number of bids and add prefix
    bbainfo_bots.columns = map(lambda x: 'bba_' + str(x),
                               range(1, bbainfo_bots.shape[1] + 1))

    return bbainfo_bots


def gather_country_info(bids_data):
    """
    Gather the country infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    bidders_country_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of occurences of each country by this bidder
        bidders_country_info.append(bids['country'].value_counts())

        pd.concat([pd.DataFrame(bidders_country_info[0]).transpose(),
                   pd.DataFrame([bidder_ids[0]])], axis=1)

    bc_info = pd.concat(bidders_country_info, axis=1).transpose()
    bidders_country_info \
        = pd.concat([bc_info,
                     pd.DataFrame(bidder_ids, columns=['bidder_id'])], axis=1)\
        .set_index('bidder_id')

    bidders_country_info.fillna(0)

    return bidders_country_info


def gather_device_info(bids_data):
    """
    Gather the device infromation from bids data.
    """

    bidder_ids = bids_data['bidder_id'].unique()
    num_bidders = len(bidder_ids)

    bidders_device_info = []
    # for each bidder
    for i in range(num_bidders):
        if i % 10 == 0:
            print "%d/%d" % (i, num_bidders)
        # get bids by this bidder
        bids = bids_data[bids_data['bidder_id'] == bidder_ids[i]]

        # number of occurences of each device by this bidder
        bidders_device_info.append(bids['device'].value_counts())

        pd.concat([pd.DataFrame(bidders_device_info[0]).transpose(),
                   pd.DataFrame([bidder_ids[0]])], axis=1)

    bd_info = pd.concat(bidders_device_info, axis=1).transpose()
    bidders_device_info \
        = pd.concat([bd_info,
                     pd.DataFrame(bidder_ids, columns=['bidder_id'])], axis=1)\
        .set_index('bidder_id')

    bidders_device_info.fillna(0)

    return bidders_device_info
