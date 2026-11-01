"""Train-only transformations and shuffled K-folds over unique input groups."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import FEATURES, FOLDS, SEED, TEST_SIZE


def make_pipeline(degree=1):
    if degree not in (1, 2, 3):
        raise ValueError("Supported degrees are 1, 2 and 3")
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale_inputs", StandardScaler()),
                     ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("scale_terms", StandardScaler()),
                     ("regression", LinearRegression())])


def make_splits(frame, folds=FOLDS, seed=SEED):
    if folds not in (3, 5, 10):
        raise ValueError("Choose 3, 5 or 10 folds")
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()
    unique = pd.unique(groups)
    train_groups, test_groups = train_test_split(unique, test_size=TEST_SIZE, random_state=seed)
    train = np.flatnonzero(np.isin(groups, train_groups))
    test = np.flatnonzero(np.isin(groups, test_groups))
    local_groups = groups[train]
    cv = []
    for fit, val in KFold(n_splits=folds, shuffle=True, random_state=seed).split(train_groups):
        cv.append((np.flatnonzero(np.isin(local_groups, train_groups[fit])),
                   np.flatnonzero(np.isin(local_groups, train_groups[val]))))
    return train, test, cv, groups
