import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt

from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    #how many days to make or lose x percent
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # load percentages
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.025
    for col in cols:
        if col > requirement+.002:
            return 1
        elif col < -requirement-.003:
            return -1
    return 0

def extract_featuresets(ticker):
    hm_days = 7
    tickers, df = process_data_for_labels(ticker)
    itter = [df['{}_{}d'.format(ticker, i)] for i in range(1,hm_days+1)]
    
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *itter))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread: ', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    #percent change of each row as compaired to the previous row
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    

    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    df_vals.plot()
    plt.show()
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25)
    # throws all three at problem and classifiers vote on classification
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    # train model
    clf.fit(X_train, y_train)
    # accuracy of models using test data
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    predictions = clf.predict(X_test)


    print('Predicted spread: ', Counter(predictions))

    return confidence

do_ml('AAPL')