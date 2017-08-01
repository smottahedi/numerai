"""
Build NNs with 2-fold statified cross validation
"""

import pandas as pd
import zipfile
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras.optimizers import Adam, SGD, Adamax
from keras.constraints import max_norm
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer
from sklearn.decomposition import PCA
from sklearn import preprocessing
from keras.layers.noise import GaussianNoise
import numpy as np

def load_data():

    zip_ref = zipfile.ZipFile('data/numerai_datasets.zip', 'r')
    zip_ref.extractall('data/')
    zip_ref.close()
    data = pd.read_csv('data/numerai_training_data.csv')
    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    indx = ~val_data.target.isnull()
    X_val = val_data.iloc[:, 3:-1]
    x = X_val.loc[indx]
    y = val_data.target.loc[indx]
    features = data.iloc[:, 3:-1]
    features = np.concatenate((features.values, x.values), axis=0)
    labels = data.iloc[:, -1]
    labels = np.concatenate((labels.values, y.values))
    era = data['era'].values
    era = np.concatenate((era, val_data.era.loc[indx]), axis=0)
    return (features, labels, era,
            X_val.values, val_data.id.values)


def create_model(activation='tanh', learning_rate=0.01, optimizer='adamax',
                 dropout=0.25, L1=0.0000005, L2=0.0000005,
                 w1_len=600, w2_len=600, w3_len=600,
                 w4_len=600, w5_len=600):
    model = Sequential()
    model.add(Dense(w1_len, input_dim=21,
                    kernel_regularizer=l2(L2),
                    activity_regularizer=l1(L1),
                    kernel_constraint=max_norm()))
    model.add(GaussianNoise(0.01))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(w2_len,
                    kernel_regularizer=l2(L2),
                    activity_regularizer=l1(L1),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(w3_len,
                    kernel_regularizer=l2(L2),
                    activity_regularizer=l1(L1),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(w4_len,
                    kernel_regularizer=l2(L2),
                    activity_regularizer=l1(L1),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(w5_len,
                    kernel_regularizer=l2(L2),
                    activity_regularizer=l1(L1),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    optimizers = {'adam': Adam(lr=learning_rate, decay=1e-6),
                  'sgd': SGD(lr=0.005, decay=1e-6,
                             momentum=0.9, nesterov=True),
                  'adamax': Adamax(decay=1e-6)}

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers[optimizer],
                  metrics=['accuracy'])

    return model


def train_and_evaluate__model(model, features, labels, trainIndx, testIndx,
                              epochs=50, batch_size=64):
    model.fit(features[trainIndx], labels[trainIndx],
              validation_data=(features[testIndx], labels[testIndx]),
              epochs=epochs, batch_size=batch_size, verbose=0)


if __name__ == '__main__':

    n_folds = 2
    features, labels, era, x_vals, val_ids = load_data()

    x = preprocessing.StandardScaler().fit_transform(features)

    pca = PCA(n_components=21, whiten=1)
    features = pca.fit_transform(x)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    skf.get_n_splits(features, era)
    n_models = 10
    log_loss_scorer = make_scorer(log_loss, needs_proba=True)

    params = {'activation': ['tanh'],
              'optimizer': ['adamax'],
              'dropout': [0.5],
              'w1_len': [128],
              'w2_len': [128],
              'w3_len': [128],
              'w4_len': [128],
              'w5_len': [182],
              'L1': [0.000005],
              'L2': [0.00005],
              'learning_rate': [0.001],
              'batch_size': [128]}

    predictions = []
    for j in range(n_models):
        # model = KerasClassifier(build_fn=create_model, verbose=1)
        model = create_model()
        for i, (trainIndx, testIndx) in enumerate(skf.split(features, era)):
            print("model:", j, ", Running Fold, ", i+1, "/", n_folds)
            train_and_evaluate__model(model, features,
                                      labels, trainIndx, testIndx)
            # if i > 0:
            #     break
            # try:
            #     grid = GridSearchCV(estimator=model, param_grid=params,
            #                         n_jobs=6, scoring=log_loss_scorer)
            #     grid_result = grid.fit(features[trainIndx], labels[trainIndx])
            # except:
            #     pass

            # means = grid_result.cv_results_['mean_test_score']
            # stds = grid_result.cv_results_['std_test_score']
            # params = grid_result.cv_results_['params']
            # for mean, stdev, param in zip(means, stds, params):
            #     print("%f (%f) with: %r" % (mean, stdev, param))

        # grid.fit(features[trainIndx], labels[trainIndx])
        # predictions.append(grid.predict_proba(x_vals, verbose=1))
        predictions.append(model.predict_proba(x_vals, verbose=1))
    with open('./predictions/nn_' + str(datetime.now()) + '.pkl', 'wb') as f:
        pickle.dump(predictions, f)
