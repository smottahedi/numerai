import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pickle
from datetime import datetime


if __name__ == '__main__':

    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    data = pd.read_csv('data/numerai_training_data.csv')
    X = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    era = data.era.values.flatten()
    X_val = val_data.iloc[:, 3:-1]
    clf = LogisticRegression(n_jobs=6)

    n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,)
    n_models = 10

    predictions = []
    for j in range(n_models):
        for i, (trainIndx, testIndx) in enumerate(skf.split(X, era)):
            print("model:", j, ", Running Fold, ", i+1, "/", n_folds)
            clf.fit(X.iloc[trainIndx, :], y.iloc[trainIndx])
            print('score:', clf.score(X.iloc[testIndx, :], y.iloc[testIndx]))
        predictions.append(clf.predict_proba(X_val)[:, 1])

    with open('./predictions/knn_' + str(datetime.now()) + '.pkl', 'wb') as f:
        pickle.dump(predictions, f)
