import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, make_scorer
from time import time
import pickle
from datetime import datetime


if __name__ == '__main__':

    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    data = pd.read_csv('data/numerai_training_data.csv')
    X = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    era = data.era.values.flatten()
    X_val = val_data.iloc[:, 3:-1]
    clf = svm.SVC(probability=True)
    log_loss_scorer = make_scorer(log_loss, needs_proba=True)

    param_dist = {"kernel": ['rbf'],
                  "gamma": [5]}

    n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    n_models = 1

    predictions = []
    for j in range(n_models):
        for i, (trainIndx, testIndx) in enumerate(skf.split(X, era)):
            print("model:", j, ", Running Fold, ", i+1, "/", n_folds)
            n_iter_search = 1
            random_search = RandomizedSearchCV(clf,
                                               param_distributions=param_dist,
                                               n_iter=n_iter_search,
                                               scoring=log_loss_scorer)

            start = time()
            random_search.fit(X.iloc[trainIndx, :], y.iloc[trainIndx])
        predictions.append(random_search.predict_proba(X_val)[:, 1])

    with open('./predictions/svm_' + str(datetime.now()) + '.pkl', 'wb') as f:
        pickle.dump(predictions, f)
