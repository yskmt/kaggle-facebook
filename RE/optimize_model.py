"""
optimize_model.py

Facebook Recruiting IV: Human or Robot?

author: Yusuke Sakamoto

"""

import sys
from sys import argv
import json
import numpy as np
import pandas as pd

from sklearn.metrics import auc, roc_curve
from sklearn import cross_validation

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import xgboost as xgb

import fb_funcs
from fb_funcs import predict_cv, fit_and_predict
from utils import (append_merchandises, append_countries, append_bba,
                   append_devices, append_bids_intervals, append_info,
                   write_submission)
# Under/Over sampling routines
# https://github.com/fmfn/UnbalancedDataset/blob/master/Notebook_UnbalancedDataset.ipynb
ubd_path = '/home/ubuntu/src/UnbalancedDataset'
if ubd_path not in sys.path:
    sys.path.append(ubd_path)

from UnbalancedDataset import SMOTE, SMOTETomek, SMOTEENN


##########################################################################
# Load prprocessed data
##########################################################################
if len(argv) < 2:
    print "sepcify the preprocessed data file #!!"
    sys.exit()
    
else:
    n_resume = float(argv[1])

    print "loading preprocessed data..."
    info_humans = pd.read_csv('data/data_pp/info_humans_%d.csv' % (n_resume),
                              index_col=0)
    info_bots = pd.read_csv('data/data_pp/info_bots_%d.csv' % (n_resume),
                            index_col=0)
    info_test = pd.read_csv('data/data_pp/info_test_%d.csv' % (n_resume),
                            index_col=0)

    print info_test.describe()
    keys_use = info_humans.keys()
    feature_set = pd.DataFrame(keys_use, columns=['features'])


############################################################################
# optimizer
############################################################################


def xgb_objective(params):
    print params

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
    X_train = info_given.sort(axis=1).as_matrix()
    y_train = labels_train
    X_test = info_test.sort(axis=1).as_matrix()

    features = info_given.sort(axis=1).keys()

    # Xgboost!
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

    cv = 5
    cv_result = xgb.cv(xgb_params, dtrain, num_rounds, nfold=cv,
                       metrics={'rmse', 'error', 'auc'}, seed=0)

    auc_max = np.max(np.array(
        map(lambda x: float(x.split('\t')[1].split(':')[1].split('+')[0]), cv_result)))
    ind_max = np.argmax(np.array(
        map(lambda x: float(x.split('\t')[1].split(':')[1].split('+')[0]), cv_result)))
    std_max = float(
        cv_result[ind_max].split('\t')[1].split(':')[1].split('+')[1])

    # logging
    with open('log_results_1.txt', 'a') as f:
        f.write(str({'loss': auc_max, 'std': std_max, 'status':
                     STATUS_OK, 'ind': ind_max}))
        f.write('\n')

    with open('log_params_1.txt', 'a') as f:
        f.write(str(params))
        f.write('\n')

    return {'loss': -auc_max, 'std': std_max, 'status': STATUS_OK, 'ind': ind_max}


def optimize_xgb(trials):
    space = {
        'num_rounds': 5000,
        'eta': hp.choice('eta', [0.001, 0.002]),
        'gamma': hp.quniform('gamma', 0.5, 10, 0.5),
        'max_depth': hp.quniform('max_depth', 4, 13, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.05, 1, 0.05),
        'nthread': 8,
        'silent': 1
    }

    best = fmin(xgb_objective, space,
                algo=tpe.suggest, trials=trials, max_evals=100)

    # logging
    with open('trials_results_1.txt', 'w') as f:
        json.dump(trials.results, f)
    with open('trials_trials_1.txt', 'w') as f:
        json.dump(trials.trials, f)

    print best


def objective(params):
    """
    prediction by ensemble objective

    """

    n_folds = params['n_folds']
    smote = params['smote']
    
    X, y, features, scaler = fb_funcs.get_Xy(info_humans, info_bots,
                                             scale=params['scale'])

    # split for cv
    kf = cross_validation.StratifiedKFold(
        y, n_folds=n_folds, shuffle=True, random_state=None)

    # cv scores
    roc_auc = np.zeros([n_folds])

    n_cv = 0
    for train_index, test_index in kf:
        print "CV#: ", n_cv
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if smote:
            ratio = float(np.count_nonzero(y==0))/float(np.count_nonzero(y==1))

            if smote=='enn':
                _smote = SMOTEENN(ratio=ratio, verbose=True)
            elif smote=='tomek':
                _smote = SMOTETomek(ratio=ratio, verbose=True)
            else:
                _smote = SMOTE(ratio=ratio, verbose=True, kind=smote)

            try:
                X_train, y_train = _smote.fit_transform(X_train, y_train)
            except:
                return {'loss': 0.0, 'std': 0.0,
                        'status': STATUS_FAIL}
            
            # shuffle just in case
            index_sh = np.random.choice(len(y_train), len(y_train), replace=False)
            X_train = X_train[index_sh]
            y_train = y_train[index_sh]
        
        result_pp = fb_funcs.predict_proba(X_train, y_train, X_test, params)
        ytps = result_pp['y_test_proba']
        fpr, tpr, thresholds = roc_curve(y_test, ytps)
        roc_auc[n_cv] = auc(fpr, tpr)

        # validation errors
        fpr, tpr, thresholds = roc_curve(y_test, ytps)
        roc_auc[n_cv] = auc(fpr, tpr)

        n_cv += 1

    # logging
    with open('log_results_%s.txt' % params['model'], 'a') as f:
        f.write(str({'loss': roc_auc.mean(), 'std': roc_auc.std(),
                     'status': STATUS_OK, }))
        f.write('\n')

    with open('log_params_%s.txt' % params['model'], 'a') as f:
        f.write(str(params))
        f.write('\n')

    return {'loss': -roc_auc.mean(), 'std': roc_auc.std(),
            'status': STATUS_OK}


def optimize(trials):
    space_lr = {
        'model': 'logistic',
        'penalty': hp.choice('penalty', ['l2', 'l1']),
        'C': hp.choice('C', 10.0**np.array(range(-5, 5))),
        'scale': hp.choice('scale', ['standard', 'log']),
        'class_weight': hp.choice('class_weight', ['auto', None]),
        'smote': hp.choice('smote',
                           [None, 'regular', 'borderline1', 'borderline2', 'svn',
                            'enn', 'tomek']),
        'n_folds': 5
    }

    space_svc = {
        'model': 'SVC',
        'gamma': hp.choice('gamma', 10.0**np.array(range(-5, 5))),
        'C': hp.choice('C', 10.0**np.array(range(-5, 5))),
        'scale': hp.choice('scale', ['standard', 'log']),
        'class_weight': hp.choice('class_weight', ['auto', None]),
        'smote': hp.choice('smote',
                           [None, 'regular', 'borderline1', 'borderline2', 'svn',
                            'enn', 'tomek']),
        'n_folds': 5
    }

    space_kn = {'model': 'KN',
                'n_neighbors': hp.choice('n_neighbors', 2**np.array(range(1, 9))),
                'weights': hp.choice('weights', ['distance', 'uniform']),
                'metric': hp.choice('metric', ['minkowski', 'manhattan']),
                'algorithm': 'auto',
                'smote': hp.choice('smote',
                   [None, 'regular', 'borderline1', 'borderline2', 'svn',
                    'enn', 'tomek']),
                'scale': hp.choice('scale', ['standard', 'log']),
                'n_folds': 5
                }

    space_et = {'model': 'ET',
                'n_estimators': hp.choice('n_estimators', [1000, 2000]),
                'max_features': hp.choice('max_features', ['auto', None, 0.5, 0.25, 0.125]),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'plot_importance': False,
                'verbose': 1,
                'n_jobs': -1,
                'max_depth': hp.choice('max_depth', [None, 2, 4, 8]),
                'class_weight': hp.choice('class_weight', ['auto', None]),
                'smote': hp.choice('smote',
                                   [None, 'regular', 'borderline1', 'borderline2', 'svn',
                                    'enn', 'tomek']),
                'scale': None,
                'n_folds': 5
                }

    space_rf = {'model': 'RF',
                'n_estimators': hp.choice('n_estimators', [1000, 2000]),
                'max_features': hp.choice('max_features', ['auto', None, 0.5, 0.25, 0.125]),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'plot_importance': False,
                'verbose': 1,
                'n_jobs': -1,
                'max_depth': hp.choice('max_depth', [None, 2, 4, 8]),
                'scale': None,
                'n_folds': 5
                }

    best = fmin(objective, space_et,
                algo=tpe.suggest, trials=trials, max_evals=300)

    # logging
    with open('trials_results_1.txt', 'w') as f:
        json.dump(trials.results, f)
    with open('trials_trials_1.txt', 'w') as f:
        json.dump(trials.trials, f)

    print best


# Trials object where the history of search will be stored
trials = Trials()

best = optimize(trials)

print best
