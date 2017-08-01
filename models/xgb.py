import pandas as pd
import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from time import time
import pickle
from datetime import datetime
from sklearn.metrics import log_loss, make_scorer


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


if __name__ == '__main__':

    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    data = pd.read_csv('data/numerai_training_data.csv')
    X = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    era = data.era.values.flatten()
    X_val = val_data.iloc[:, 3:-1].values
    clf = XGBoostClassifier(
        eval_metric = 'logloss',
        num_class = 2,
        nthread = 4,
        eta = 0.1,
        num_boost_round = 80,
        max_depth = 12,
        subsample = 0.5,
        colsample_bytree = 1.0,
        silent = 1)
    log_loss_scorer = make_scorer(log_loss, needs_proba=True)

    parameters = {'num_boost_round': [100, 250, 500],
                'eta': [0.05, 0.01, 0.001],
                'max_depth': [6, 9, 12],
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.9, 1.0]}

    n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,)
    n_models = 10

    predictions = []
    for j in range(n_models):
        for i, (trainIndx, testIndx) in enumerate(skf.split(X, era)):
            print("model:", j, ", Running Fold, ", i+1, "/", n_folds)
            n_iter_search = 5
            random_search = RandomizedSearchCV(clf,
                                               param_distributions=parameters,
                                               n_iter=n_iter_search,
                                               scoring=log_loss_scorer)

            start = time()
            random_search.fit(X.iloc[trainIndx, :].values,
                               y.iloc[trainIndx].values)

        best_parameters, score, _ = max(random_search.grid_scores_, key=lambda x: x[1])
        print(score)
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        predictions.append(random_search.
                           predict_proba((X_val))[:, 1])

    with open('./predictions/xgb_' + str(datetime.now()) + '.pkl', 'wb') as f:
        pickle.dump(predictions, f)
