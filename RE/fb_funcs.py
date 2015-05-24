import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

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

def predict_cv(info_humans, info_bots, plot_roc=False, n_folds=5,
               params=None):
    """
    prediction by undersampling

    p_valid: validation set fraction
    """

    model = params['model']
    
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

    n_cv = 0
    for train_index, test_index in kf:
        print "CV#: ", n_cv
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if "ET" in model:
            # clf = SGDClassifier(loss='log')
            # clf = DecisionTreeClassifier()
            # clf = KNeighborsClassifier()
            # clf = RandomForestClassifier(n_estimators=n_estimators,
            # class_weight=None, max_features=None)
            # clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.1)

            clf = ExtraTreesClassifier(n_estimators=params['n_estimators'],
                                       n_jobs=params['n_jobs'],
                                       max_features=params['max_features'],
                                       criterion=params['criterion'],
                                       verbose=params['verbose'],
                                       random_state=0)
            clf.fit(X_train, y_train)
            y_test_proba = clf.predict_proba(X_test)
            y_test_pred = clf.predict(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
            roc_auc[n_cv] = auc(fpr, tpr)
            clf_score[n_cv] = clf.score(X_test, y_test)

        elif 'XGB' in model:
            # XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            xgb_params = {"objective": "binary:logistic",
                          'eta': params['eta'],
                          'gamma': params['gamma'],
                          'max_depth': params['max_depth'],
                          'min_child_weight': params['min_child_weight'],
                          'subsample': params['subsample'],
                          'colsample_bytree': params['colsample_bytree'],
                          'nthread': params['nthread'],
                          'silent': params['silent']}
            num_rounds = int(params['num_rounds'])

            evallist = [(dtrain, 'train')]
            bst = xgb.train(xgb_params, dtrain, num_rounds, evallist)
            dtest = xgb.DMatrix(X_test)
            y_test_proba = bst.predict(dtest)
            fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
            roc_auc[n_cv] = auc(fpr, tpr)
            clf_score[n_cv] = 0.0
            clf = 0
            
        if plot_roc:
            # Plot ROC curve
            # plt.clf()
            plt.plot(fpr, tpr, label='ROC curve # %d (area = %0.2f)' %
                     (n_cv, roc_auc[n_cv]))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

        n_cv += 1

    return clf, roc_auc, clf_score


def fit_and_predict(info_humans, info_bots, info_test,
                    params, p_use=None):

    model = params['model']

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
        clf = RandomForestClassifier(n_estimators=n_estimators, verbose=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)
        return y_pred[:, 1], y_train_pred[:, 1], 0

    elif model == 'KN':
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)
        return y_pred[:, 1], y_train_pred[:, 1], 0, 0
        
    elif model == 'ET':
        clf = ExtraTreesClassifier(n_estimators=params['n_estimators'],
                                   n_jobs=params['n_jobs'],
                                   max_features=params['max_features'],
                                   criterion=params['criterion'],
                                   verbose=params['verbose'],
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

        print list((features[indices]))[:40]

        # Plot the feature importances of the forest
        if params['plot_importance']:
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
    elif "XGB" in model:
        xgb_params = {"objective": "binary:logistic",
                      'eta': params['eta'],
                      'gamma': params['gamma'],
                      'max_depth': params['max_depth'],
                      'min_child_weight': params['min_child_weight'],
                      'subsample': params['subsample'],
                      'colsample_bytree': params['colsample_bytree'],
                      'nthread': params['nthread'],
                      'silent': params['silent']}
        num_rounds = int(params['num_rounds'])
        
        dtrain = xgb.DMatrix(X_train, label=y_train)

        if "CV" in model:
            cv=5
            cv_result = xgb.cv(xgb_params, dtrain, num_rounds, nfold=cv,
                               metrics={'rmse', 'error', 'auc'}, seed=0)
            return 0, 0, cv_result, 0
        else:
            evallist = [(dtrain, 'train')]
            bst = xgb.train(xgb_params, dtrain, num_rounds, evallist)
            dtest = xgb.DMatrix(X_test)
            y_pred = bst.predict(dtest)
            y_train_pred = bst.predict(dtrain)

            return y_pred, y_train_pred, y_train, 0


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

def append_info(info, info_new, keys_appended):
    """
    Append info
    """

    info_appended = []
    for key in keys_appended:
        if key in list(info_new.keys()):
            info_appended.append(info_new[key])
        else:
            # just create zero-column
            info_appended.append(
                pd.DataFrame(np.zeros(len(info_new)),
                             index=info_new.index, columns=[key]))

    info_appended.append(info)

    info = pd.concat(info_appended, axis=1)

    return info


    
