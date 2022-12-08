from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
import time

class CustomKNNRegressor(BaseEstimator):

    def __init__(self, k, distance_metric='euclidean', aggregation_fn=np.mean):
        self.k = k
        self.distance_metric = distance_metric
        self.euclidean_distance = lambda a, b: np.sqrt(np.sum(
            (a - b)**2))  # faster than iterating over the values and adding them up (uses optimizations offered by NP)
        self.manhattan_distance = lambda a, b: np.sum(np.abs(a - b))
        self.aggregation_fn = aggregation_fn

    def _get_distance_metric(self):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance
        else:
            raise Exception("Invalid Distance metric")

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def _predict_one(self, row):
        distance_metric_fn = self._get_distance_metric()
        distances = self.X_train.apply(lambda x: self.euclidean_distance(x, row), axis=1)
        # finding the indices of k smallest distances
        notInYtrain = []
        for distance in distances.index:
            if distance not in self.y_train.index:
                notInYtrain.append(distance)


        idx = np.argpartition(distances, self.k)[:self.k]

        #idx = distances.sort_values()[:self.k].index

        # print(
        #     f"Smallest indices: {idx}\n\t Those ys are: {self.y_train.iloc[idx]}\n\t aggregated result: {self.aggregation_fn(self.y_train.iloc[idx])}")
        return self.aggregation_fn(self.y_train.iloc[idx])

    def predict(self, X_test):
        return X_test.apply(lambda x: self._predict_one(x), axis=1)


def custom_knn_grid_search_cv(model, ks=[3], distance_metrics=['euclidean'], cv=5):
    best_estimator = None
    best_score = -np.infty

    for k in ks:
        for metric in distance_metrics:
            model = CustomKNNRegressor(k, distance_metric=metric)
            k_folds = KFold(n_splits=cv)
            cv_scores = cross_val_score(model, X_train, y_train,
                                        scoring=make_scorer(lambda y_true, y_pred: np.mean( (y_true-y_pred) **2 ) ,
                                                            greater_is_better=False),
                                        cv=k_folds)
            cv_scores_mean = cv_scores.mean()
            if cv_scores_mean > best_score:
                best_score = cv_scores_mean
                best_estimator = model

    return {
        "best_estimator": best_estimator,
        "best_score": best_score
    }


if __name__ == '__main__':
    comp_hw_df = pd.read_csv("Data/machine.data", header=None)
    comp_hw_df.columns = ['vendor_name', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

    # dropping the model column even though it could be useful
    comp_hw_df = comp_hw_df.drop(columns=['model'])

    comp_hw_df = pd.get_dummies(comp_hw_df)



    X = comp_hw_df.drop(columns=['ERP'])
    y = comp_hw_df.loc[:, 'ERP']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    myKnn = CustomKNNRegressor(3)


    knn_params = {
        'k': [2, 3],
        'distance_metric': ['euclidean', 'manhattan'],
    }

    result = custom_knn_grid_search_cv(myKnn, ks=[2,3], distance_metrics=['euclidean', 'manhattan'])






