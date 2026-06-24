import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

def select_features(X, y):
    selector = SelectKBest(mutual_info_classif, k=10)
    X_new = selector.fit_transform(X, y)
    return X_new
