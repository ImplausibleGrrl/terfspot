# christina lu
# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from joblib import dump, load

train_following = '~/cs/thesis/features/all_following.csv'
labels = '~/cs/thesis/features/labels.csv'

train_all = '~/cs/thesis/models/proba_all.csv'

def train_following_log_reg():
    df1 = pd.read_csv(train_following)
    df2 = pd.read_csv(labels)

    df = pd.merge(df1, df2, on='user_id')

    X = df.drop(['user_id', 'label'], axis=1).values
    y = df['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model

def train_overall_log_reg():
    df = pd.read_csv(train_all)

    X = df.drop(['user_id', 'label', 'topic_pred'], axis=1).values
    y = df['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model
#
# model = train_following_log_reg()
# dump(model, 'following_logreg_model.joblib')
