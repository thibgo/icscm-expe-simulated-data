"""
    Invariant Causal Set Covering Machine -- A invariance-oriented version of the Set Covering Machine in Python
    Copyright (C) 2023 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from six import iteritems

import math
import logging
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

import warnings
from itertools import chain, combinations  # for powerset
from gsq import ci_tests  # conditional independence tests
from sklearn.tree import DecisionTreeClassifier


def _class_to_string(instance):
    """
    Returns a string representation of the public attributes of a class.

    Parameters:
    -----------
    instance: object
        An instance of any class.

    Returns:
    --------
    string_rep: string
        A string representation of the class and its public attributes.

    Notes:
    -----
    Private attributes must be marked with a leading underscore.
    """
    return (
        instance.__class__.__name__
        + "("
        + ",".join(
            [
                str(k) + "=" + str(v)
                for k, v in iteritems(instance.__dict__)
                if str(k[0]) != "_"
            ]
        )
        + ")"
    )


class BaseModel(object):
    def __init__(self):
        self.rules = []
        super(BaseModel, self).__init__()

    def add(self, rule):
        self.rules.append(rule)

    def predict(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def remove(self, index):
        del self.rules[index]

    @property
    def example_dependencies(self):
        return [d for ba in self.rules for d in ba.example_dependencies]

    @property
    def type(self):
        raise NotImplementedError()

    def _to_string(self, separator=" "):
        return separator.join([str(a) for a in self.rules])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        for ba in self.rules:
            yield ba

    def __len__(self):
        return len(self.rules)

    def __str__(self):
        return self._to_string()


class InvariantCausalPredictionDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        threshold=0.05,
        min_samples_split=2,
        max_depth=None,
        random_state=None,
    ):
        self.threshold = threshold
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "threshold": self.threshold,
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Find the set of causal parents of the target variable.
        Fit a SCM model on this set.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The first columns contains the environment id of the example.
            The features of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        self: object
            Returns self.

        """
        self.random_state = check_random_state(self.random_state)

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug(
            "The data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
                len(y),
                self.classes_[0],
                total_n_ex_by_class[0],
                self.classes_[1],
                total_n_ex_by_class[1],
            )
        )

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for X:", type(X))

        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for y:", type(X))

        # Create an empty model
        logging.debug("Initializing empty model")
        logging.debug("Training start")
        ones = np.ones(len(y))
        remaining_N = ones - y  # remaining negative examples
        remaining_P = y

        # first column of X: environment
        env_of_examples = X[:, 0]

        # Extract features only
        X_original_with_env = (
            X.copy()
        )  # useful for prediction for feature importance computation
        X = X[:, 1:]

        variables_idx = list(range(X_original_with_env.shape[1]))[
            1:
        ]  # because feature 0 in X_original_with_env is the environment
        sets = list(
            chain.from_iterable(
                combinations(variables_idx, r) for r in range(X.shape[1])
            )
        )
        sets_that_creates_indep = []
        conditional_indep_calculation_df = pd.DataFrame(
            X_original_with_env, columns=["e"] + variables_idx
        )
        conditional_indep_calculation_df["y"] = y
        y_position = conditional_indep_calculation_df.columns.get_loc("y")
        e_position = conditional_indep_calculation_df.columns.get_loc("e")
        for s in sets:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                p_value = ci_tests.ci_test_bin(
                    conditional_indep_calculation_df.values,
                    e_position,
                    y_position,
                    list(s),
                )
            if p_value > self.threshold:
                sets_that_creates_indep.append(s)
                print(f'set={s}, p_value={round(p_value, 6)}                         independent ')
            else:
                print(f'set={s}, p_value={round(p_value, 6)}')
        print('sets_that_creates_indep', sets_that_creates_indep)
        # intersection of sets:
        if len(sets_that_creates_indep) == 0:
            intersection_sets_that_creates_indep = []
        else:
            intersection_sets_that_creates_indep = list(
                set.intersection(*map(set, sets_that_creates_indep))
            )
        print('intersection_sets_that_creates_indep', intersection_sets_that_creates_indep)

        # Calculate classic Decision Tree training set made of only parent variables
        X_restricted_to_parents = np.zeros(X_original_with_env.shape)
        for i in intersection_sets_that_creates_indep:
            X_restricted_to_parents[:, i] = X_original_with_env[:, i]
        dt_model = DecisionTreeClassifier(
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        dt_model.fit(X_restricted_to_parents, y)
        self.model_ = dt_model

        logging.debug("Training completed")

        logging.debug("Calculating rule importances")
        if len(intersection_sets_that_creates_indep) == 0:
            self.feature_importances_ = [0] * X_original_with_env.shape[1]
        else:
            self.feature_importances_ = [
                int(i in intersection_sets_that_creates_indep)
                for i in range(X_original_with_env.shape[1])
            ]
        logging.debug("Done.")

        return self

    def predict(self, X):
        """
        Predict class

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.

        Returns:
        --------
        predictions: numpy_array, shape=[n_examples]
            The predicted class for each example.

        """
        check_is_fitted(self, ["model_", "feature_importances_", "classes_"])
        X = check_array(X)
        return self.classes_[self.model_.predict(X)]

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.

        Returns:
        --------
        p : array of shape = [n_examples, 2]
            The class probabilities for each example. Classes are ordered by lexicographic order.

        """
        warn(
            "SetCoveringMachines do not support probabilistic predictions. The returned values will be zero or one.",
            RuntimeWarning,
        )
        check_is_fitted(self, ["model_", "feature_importances_", "classes_"])
        X = check_array(X)
        pos_proba = self.classes_[self.model_.predict(X)]
        neg_proba = 1.0 - pos_proba
        return np.hstack((neg_proba.reshape(-1, 1), pos_proba.reshape(-1, 1)))

    def score(self, X, y):
        """
        Predict classes of examples and measure accuracy

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        accuracy: float
            The proportion of correctly classified examples.

        """
        check_is_fitted(self, ["model_", "feature_importances_", "classes_"])
        X, y = check_X_y(X, y)
        return accuracy_score(y_true=y, y_pred=self.predict(X))

    def _append_conjunction_model(self, new_rule):
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _append_disjunction_model(self, new_rule):
        new_rule = new_rule.inverse()
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _get_example_idx_by_class_conjunction(self, y):
        positive_example_idx = np.where(y == 1)[0]
        negative_example_idx = np.where(y == 0)[0]
        return positive_example_idx, negative_example_idx

    def _get_example_idx_by_class_disjunction(self, y):
        positive_example_idx = np.where(y == 0)[0]
        negative_example_idx = np.where(y == 1)[0]
        return positive_example_idx, negative_example_idx

    def __str__(self):
        return _class_to_string(self)
