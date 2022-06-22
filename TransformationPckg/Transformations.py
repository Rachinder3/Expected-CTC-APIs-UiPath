import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class Drop_undesired_features(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=['IDX', 'Applicant_ID']):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self  # nothing to do

    def transform(self, X, y=None):
        df = X.copy()
        for feature in self.features_to_drop:
            ## try and catch to avoid run time errors
            try:
                df.drop(feature, axis=1, inplace=True)
            except:
                pass

        return df


class Handle_Categorical_Features(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_features = []  # empty categorical features list

    def fit(self, X, y=None):
        try:
            # Get the categorical features
            self.categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
        except:
            pass

        return self

    def transform(self, X, y=None):
        ## standardize the features
        df = X.copy()
        try:
            for feature in self.categorical_features:
                df[feature] = df[feature].str.lower()
        except:
            pass

        ## Handle missing values
        try:
            df[self.categorical_features] = df[self.categorical_features].fillna('missing')

        except:
            pass

        return df


class Handle_Missing_Numerical_Features(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_features_nan = []  # empty numerical features list
        self.medians = []

    def fit(self, X, y=None):
        try:
            # Get the numerical features
            self.numerical_features_nan = [feature for feature in X.columns if
                                           X[feature].dtype != 'O' and X[feature].isnull().sum() >= 1]

            self.medians = []
            for feature in self.numerical_features_nan:
                self.medians.append(X[feature].median())

        except:
            pass
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for index, feature in enumerate(self.numerical_features_nan):

            try:
                median_value = self.medians[index]
                ## create new feature to capture nan values
                df[feature + '_nan'] = np.where(df[feature].isnull(), 1, 0)
                ### replace the missing values with median
                df[feature] = df[feature].fillna(median_value)

            except:
                pass
        return df


class handle_temporal_features(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.temporal_features = []

    def fit(self, X, y=None):
        try:
            # get the temporal features
            self.temporal_features = [feature for feature in X.columns if 'year' in feature.lower()]
        except:
            pass

        return self

    def transform(self, X, y=None):
        df = X.copy()
        for feature in self.temporal_features:
            ### extract numbers of years that have passed from when a person completed a particular degree
            try:
                new_feature_name = "{}-{}".format(datetime.datetime.now().year, feature)
                df[new_feature_name] = datetime.datetime.now().year - df[feature]
                df[new_feature_name] = df[new_feature_name].astype(int)
                df.drop(feature, axis=1, inplace=True)
            except:
                pass
        return df


class OHE_Categorical_Features(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_features = []
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    def fit(self, X, y=None):
        try:
            self.categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
            self.ohe.fit(X[self.categorical_features])
            ##print(self.ohe.categories_)
        except:
            pass

        return self

    def transform(self, X, y=None):
        df = X.copy()
        try:
            cat_ohe = self.ohe.transform(df[self.categorical_features])

            ohe_df = pd.DataFrame(cat_ohe, columns=self.ohe.get_feature_names(input_features=self.categorical_features),
                                  index=df.index)
            df = pd.concat([df, ohe_df], axis=1).drop(columns=self.categorical_features, axis=1)
        except Exception as e:
            pass

        return df


