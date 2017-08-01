import pickle
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam, Adamax
from keras.constraints import max_norm
from keras.layers import BatchNormalization
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import numpy as np


if __name__ == '__main__':

    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    indx = ~val_data.target.isnull()
    target = val_data.loc[indx, :].values
    X = val_data.iloc[:, 3:-1].values

    predictions = {}
    j = 0

    for fname in os.listdir('./predictions/'):
        f = open('./predictions/' + fname, 'rb')
        pred = pickle.load(f)
        for i in range(len(pred)):
            predictions[j] = pred[i].flatten()
            j += 1
        f.close()

    predictions = pd.DataFrame(predictions).values
    # predictions = np.concatenate((X, predictions.values), axis=1)

    input_dim = predictions.shape[1]

    model = Sequential()
    model.add(Dense(600, input_dim=input_dim,
                    kernel_regularizer=l2(0.000005),
                    activity_regularizer=l1(0.0000005),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(600,
                    kernel_regularizer=l2(0.000005),
                    activity_regularizer=l1(0.0000005),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(600,
                    kernel_regularizer=l2(0.000005),
                    activity_regularizer=l1(0.0000005),
                    kernel_constraint=max_norm()))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001)
    adamax = Adamax()

    model.compile(loss='binary_crossentropy',
                  optimizer=adamax,
                  metrics=['accuracy'])

    model.fit(predictions[indx, :],  val_data.target[indx],
              epochs=150, batch_size=64, verbose=1)

    probs = model.predict_proba(predictions, verbose=1)

    submit = pd.DataFrame({'id': val_data.id,
                           'probability': probs.flatten().round(5)})
    submit.to_csv('./submissions/submit' +
                  str(datetime.now()) + '.csv', index=False)
