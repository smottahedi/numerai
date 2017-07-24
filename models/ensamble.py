import numpy as np
import pickle
import os
import pandas as pd




if __name__ == '__main__':

    val_data = pd.read_csv('data/numerai_tournament_data.csv')
    predictions = {}
    j = 0

    for fname in os.listdir('./predictions/'):
        f = open('./predictions/' + fname, 'rb')
        pred = pickle.load(f)
        for i in range(len(pred)):
            predictions[j] = pred[i].flatten()
            j += 1
        f.close()

    predictions = pd.DataFrame(predictions).T

    probs = np.power(np.prod(predictions.values, axis=0),
                     1.0 / predictions.shape[0])



    submit = pd.DataFrame({'id':val_data.id, 'probability': probs.flatten()})
    submit.to_csv('./submissions/submit.csv', index=False)
