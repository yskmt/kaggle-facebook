import numpy as np
import sys

model = sys.argv[1]

ll = []
with open('log_results_%s.txt' % model, 'r') as f:
    for l in f:
        ll.append(float(l.split(':')[2].split(',')[0]))

ll = np.array(ll)
print np.max(ll), np.argmax(ll)

with open('log_params_%s.txt' % model, 'r') as f:
    print f.readlines()[np.argmax(ll)]
with open('log_results_%s.txt' % model, 'r') as f:
    print f.readlines()[np.argmax(ll)]


# logistic
# {'n_folds': 5, 'penalty': 'l1', 'C': 0.1, 'scale':'log', 'model': 'logistic'}
# {'std': 0.015756513001710624, 'loss': 0.91683053204855958, 'status':
#  'ok'}

# SVC
# {'n_folds': 5, 'C': 100.0, 'scale': 'log', 'model': 'SVC', 'gamma': 0.001}
# {'std': 0.043199539583259561, 'loss': 0.90891827015900317, 'status': 'ok'}

# KN
# {'n_neighbors': 32, 'scale': 'log', 'weights': 'distance', 'algorithm': 'auto', 'n_folds': 5, 'metric': 'minkowski', 'model': 'KN'}
# {'std': 0.0075366362947824174, 'loss': 0.9197278122503052, 'status': 'ok'}

# ET
# {'n_jobs': -1, 'verbose': 1, 'max_depth': 8, 'n_estimators': 2000, 'max_features': None, 'scale': None, 'criterion': 'gini', 'n_folds': 5, 'model': 'ET', 'plot_importance': False}
# {'std': 0.02055749912306061, 'loss': 0.9418071962597393, 'status': 'ok'}

# XGB
# {'colsample_bytree': 0.85, 'silent': 1, 'num_rounds': 5000, 'nthread': 8, 'min_child_weight': 3.0, 'subsample': 0.7, 'eta': 0.002, 'max_depth': 9.0, 'gamma': 2.0}
# {'std': 0.023379, 'loss': 0.94208099999999995, 'ind': 1825, 'status': 'ok'}

# RF
# {'n_jobs': -1, 'verbose': 1, 'max_depth': 8, 'n_estimators': 2000, 'max_features': None, 'scale': None, 'criterion': 'gini', 'n_folds': 5, 'model': 'ET', 'plot_importance': False}
# {'std': 0.02055749912306061, 'loss': 0.9418071962597393, 'status': 'ok'}
