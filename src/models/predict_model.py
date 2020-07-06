# christina lu
# predict_model.py

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from train_model import *
from features.build_features import *

outname = './following_proba.csv'

def predict_following(model, save=False):
    df = build_following_features()
    X = df.drop(['user_id'], axis=1).values

    if not save:
        pred = model.predict(X)
    else:
        pred = model.predict_proba(X)

    #np.savetxt('sample_pred.csv', pred, delimiter=',')

    if not save:
        y = len(pred) * [0]
        c = confusion_matrix(pred, y)
        f1 = f1_score(pred, y)

        print(c)
        print(f1)
    else:
        df['follow_proba'] = pred[:, 1]
        df = df[['user_id', 'follow_proba']]
        df.to_csv(outname, index=False)

def predict_overall(model):
    df = pd.read_csv('proba_all.csv')
    X = df.drop(['user_id', 'label'], axis=1).values


    pred = model.predict(X)


    #np.savetxt('sample_pred.csv', pred, delimiter=',')


    y = len(pred) * [0]
    c = confusion_matrix(pred, y)

    print(c)


predict_overall(train_overall_log_reg())
#predict_following(train_following_log_reg(), save=True)
