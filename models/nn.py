"""
Build NNs with 2-fold statified cross validation
"""


import numpy as np
import pandas as pd
import scipy as sp
import zipfile
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam, Adamax
from keras.constraints import non_neg, unit_norm, max_norm
from keras.layers import BatchNormalization





def load_data():

    zip_ref = zipfile.ZipFile('data/numerai_datasets.zip', 'r')
    zip_ref.extractall('data/')
    zip_ref.close()
    data = pd.read_csv('data/numerai_training_data.csv')
    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    X_val = val_data.iloc[:, 3:-1]

    features = data.iloc[:, 3:-1]
    labels = data.iloc[:, -1]


    return features.values, labels.values, data['era'].values, X_val.values, val_data.id.values


def create_model():
    model = Sequential()
    model.add(Dense(600, input_dim=21,
             kernel_regularizer=l2(0.00005 ),
             activity_regularizer=l1(0.000005),
             kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256,
             kernel_regularizer=l2(0.00005 ),
             activity_regularizer=l1(0.000005),
             kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256,
             kernel_regularizer=l2(0.00005 ),
             activity_regularizer=l1(0.000005),
             kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001)
    adamax = Adamax()

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model



def train_and_evaluate__model(model, features, labels, trainIndx, testIndx):
    model.fit(features[trainIndx], labels[trainIndx],
              validation_data=(features[testIndx], labels[testIndx]), epochs=20, batch_size=128, verbose=0)


if __name__ == '__main__':

    n_folds = 2
    features, labels, era, x_vals, val_ids = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    skf.get_n_splits(features, era)
    n_models = 10

    predictions = []
    for j in range(n_models):
        for i, (trainIndx, testIndx) in enumerate(skf.split(features, era)):
            print("model:", j, ", Running Fold, ", i+1, "/", n_folds)
            model = None # Clearing the NN.
            model = create_model()
            train_and_evaluate__model(model, features, labels, trainIndx, testIndx)
            predictions.append(model.predict_proba(x_vals, verbose=0))


    with open('./predictions/nn_' + str(datetime.now()) + '.pkl', 'wb') as f:
        pickle.dump(predictions, f)
