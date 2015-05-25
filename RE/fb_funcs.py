"""
fb_funcs.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation

import xgboost as xgb


def predict_cv(info_humans, info_bots, plot_roc=False, n_folds=5,
               params=None):
    """
    prediction by undersampling

    p_valid: validation set fraction
    """

    model = params['model']
    
    X, y, features = get_Xy(info_humans, info_bots)

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

        if 'XGB' in model:
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

        else:
            if "ET" in model:

                clf = ExtraTreesClassifier(n_estimators=params['n_estimators'],
                                           n_jobs=params['n_jobs'],
                                           max_features=params['max_features'],
                                           criterion=params['criterion'],
                                           verbose=params['verbose'],
                                           random_state=0)

            elif "SVC" in model:
                clf = svm.SVC(probability=True)

            elif "logistic" in model:
                clf = LogisticRegression()

            elif "RF" in model:
                clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           n_jobs=params['n_jobs'],
                                           max_features=params['max_features'],
                                           criterion=params['criterion'],
                                             verbose=params['verbose'],
                                             max_depth=params['max_depth'])
                
            clf.fit(X_train, y_train)
            y_test_proba = clf.predict_proba(X_test)
            y_test_pred = clf.predict(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
            roc_auc[n_cv] = auc(fpr, tpr)
            clf_score[n_cv] = clf.score(X_test, y_test)

            if "RF" in model:
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

    num_models = len(params)
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

    # get matrices forms
    ytps = []
    X_train, y_train, features, scaler, X_test\
        = get_Xy(info_humans, info_bots, info_test)

    for mn in range(num_models):
        ytps.append(predict_proba(X_train, y_train, X_test, params[mn]))

    ytps = np.array(ytps)
    # ensemble!
    y_test_proba = ytps.mean(axis=0)

    return y_test_proba, ytps
        
        # if "XGB_CV" in model:
        #     cv = 5
        #     cv_result = xgb.cv(xgb_params, dtrain, num_rounds, nfold=cv,
        #                        metrics={'rmse', 'error', 'auc'}, seed=0)
        #     return 0, 0, cv_result, 0

            
        # if (model == 'ET') or (model == 'RF'):
            
        #     importances = clf.feature_importances_
        #     std = np.std([tree.feature_importances_ for tree in clf.estimators_],
        #                  axis=0)
        #     indices = np.argsort(importances)[::-1]

        #     # Print the feature ranking
        #     print("Feature ranking:")
        #     for f in range(min(len(features), 40)):
        #         print("%d. feature %d: %s = (%f)"
        #               % (f, indices[f], features[indices[f]], importances[indices[f]]))

        #     print list((features[indices]))[:40]

        #     # Plot the feature importances of the forest
        #     if params['plot_importance']:
        #         plt.figure()
        #         plt.title("Feature importances")
        #         plt.bar(range(len(features)), importances[indices],
        #                 color="r", yerr=std[indices], align="center")
        #         plt.xticks(range(len(features)), features[indices])
        #         plt.xlim([-1, len(features)])
        #         plt.show()

        #     y_pred = clf.predict_proba(X_test)
        #     y_train_pred = clf.predict_proba(X_train)
            
        #     return y_pred[:, 1], y_train_pred[:, 1], 0, list((features[indices]))

        # y_pred = clf.predict_proba(X_test)
        # y_train_pred = clf.predict_proba(X_train)

        # return y_pred[:, 1], y_train_pred[:, 1], 0, 0

def get_Xy(info_humans, info_bots, info_test=None):
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
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    y = labels_train
    features = info_given.sort(axis=1).keys()

    if info_test is not None:
        X_test = info_test.sort(axis=1).as_matrix()
        X_test = scaler.transform(X_test)

        return X, y, features, scaler, X_test
        
    return X, y, features, scaler

def predict_cv_ens(info_humans, info_bots, params,
                   n_folds=5):
    """
    prediction by ensemble

    p_valid: validation set fraction
    """

    X, y, features, scaler = get_Xy(info_humans, info_bots)
    
    # split for cv
    kf = cross_validation.StratifiedKFold(
        y, n_folds=n_folds, shuffle=True, random_state=None)

    # cv scores
    roc_auc = np.zeros([n_folds, len(params)+1])

    n_cv = 0
    for train_index, test_index in kf:
        print "CV#: ", n_cv
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ytps = []
        for mn in range(len(params)):
            ytps.append(predict_proba(X_train, y_train, X_test, params[mn]))
            fpr, tpr, thresholds = roc_curve(y_test, ytps[mn])
            roc_auc[n_cv, mn] = auc(fpr, tpr)

        ytps = np.array(ytps)
        # ensemble!
        y_test_proba = ytps.mean(axis=0)

        # validation errors
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        roc_auc[n_cv, len(params)] = auc(fpr, tpr)
        
        n_cv += 1

    return roc_auc


def predict_proba(X_train, y_train, X_test, params):

    model = params['model']

    if 'XGB' in model:
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

    elif 'ET' in model:
        # extratree
        clf = ExtraTreesClassifier(n_estimators=params['n_estimators'],
                                   n_jobs=params['n_jobs'],
                                   max_features=params['max_features'],
                                   criterion=params['criterion'],
                                   verbose=params['verbose'],
                                   random_state=None)
                
        clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)[:,1]

    elif "RF" in model:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     n_jobs=params['n_jobs'],
                                     max_features=params['max_features'],
                                     criterion=params['criterion'],
                                     verbose=params['verbose'],
                                     max_depth=params['max_depth'])

        clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)[:,1]

    elif "KN" in model:
        clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'],
                                   weights=params['weights'],
                                   algorithm=params['algorithm'])

        clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)[:,1]

    elif "SVC" in model:
        clf = SVC(C=params['C'],
                  gamma=params['gamma'],
                  probability=True)
        clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)[:,1]

        
    return y_test_proba
    
    
def kfcv_ens(info_humans, info_bots, params,
             num_cv=5, num_folds=5):
    """
    k-fold cross validation
    """

    roc_auc = []
    roc_auc_std = []

    roc_auc_0 = []
    roc_auc_std_0 = []
    roc_auc_1 = []
    roc_auc_std_1 = []
    
    for i in range(num_cv):
        ra = predict_cv_ens(info_humans, info_bots, params,
                            n_folds=num_folds)

        print ra.mean(axis=0), ra.std(axis=0)
        roc_auc.append(ra.mean(axis=0))
        roc_auc_std.append(ra.std(axis=0))

    roc_auc = np.array(roc_auc)
    roc_auc_std = np.array(roc_auc_std)

    return [roc_auc.mean(axis=0), roc_auc_std.mean(axis=0)]
