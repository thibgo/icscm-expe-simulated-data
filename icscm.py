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

import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence
import warnings

from gsq import ci_tests  # conditional independence tests

from io import StringIO


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


class ConjunctionModel(BaseModel):
    def predict(self, X):
        predictions = np.ones(X.shape[0], bool)
        for a in self.rules:
            predictions = np.logical_and(predictions, a.classify(X))
        return predictions.astype(np.uint8)

    @property
    def type(self):
        return "conjunction"

    def __str__(self):
        return self._to_string(separator=" and ")
    
class DisjunctionModel(BaseModel):
    def predict(self, X):
        predictions = np.zeros(X.shape[0], bool)
        for a in self.rules:
            predictions = np.logical_or(predictions, a.classify(X))
        return predictions.astype(np.uint8)

    @property
    def type(self):
        return "disjunction"

    def __str__(self):
        return self._to_string(separator=" or ")


class BaseRule(object):
    """
    A rule mixin class

    """

    def __init__(self):
        super(BaseRule, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the rule.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return _class_to_string(self)


class DecisionStump(BaseRule):
    """
    A decision stump is a rule that applies a threshold to the value of some feature

    Parameters:
    -----------
    feature_idx: uint
        The index of the feature
    threshold: float
        The threshold at which the outcome of the rule changes
    kind: str, default="greater"
        The case in which the rule returns 1, either "greater" or "less_equal".

    """

    def __init__(self, feature_idx, threshold, kind="greater"):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.kind = kind
        super(DecisionStump, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the decision stump.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        if self.kind == "greater":
            c = X[:, self.feature_idx] > self.threshold
        else:
            c = X[:, self.feature_idx] <= self.threshold
        return c

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        return DecisionStump(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self):
        return "X[{0:d}] {1!s} {2:.3f}".format(
            self.feature_idx, ">" if self.kind == "greater" else "<=", self.threshold
        )

def stat_test(y_vector, e_vector):
    observed = pd.crosstab(y_vector, e_vector)
    if observed.size == 0:
        raise ValueError("No data; observed has size 0.")
    
    expected = pd.DataFrame(expected_freq(observed), index=observed.index, columns=observed.columns)
    dof = float(expected.size - sum(expected.shape) + expected.ndim - 1)
    if dof == 1:
        # Adjust `observed` according to Yates' correction for continuity.
        observed = observed + 0.5 * np.sign(expected - observed)
    ddof = observed.size - 1 - dof
    stats = []
    lambda_ = 1.0
    if dof == 0:
        chi2, p_value_neg_leaf, cramer, power = 0.0, 1.0, np.nan, np.nan
    else:
        chi2, p_value_neg_leaf = power_divergence(observed, expected, ddof=ddof, axis=None, lambda_=lambda_)
    return p_value_neg_leaf

class InvariantCausalSCM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        p=1.0,
        model_type="conjunction",
        max_rules=10,
        resample_rules=False,
        threshold=0.05,
        threshold_correction=None,
        pruning=True,
        stopping_method="no_more_negatives",
        random_state=None,
    ):
        self.p = p
        self.model_type = model_type
        if model_type not in ["conjunction", "disjunction"]:
            raise ValueError(
                "wrong model_type: {}, only conjunction is supported".format(model_type)
            )
        self.max_rules = max_rules
        self.resample_rules = resample_rules
        self.threshold = threshold
        self.threshold_correction = threshold_correction
        self.pruning = pruning
        self.stopping_method = stopping_method
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "model_type": self.model_type,
            "max_rules": self.max_rules,
            "resample_rules": self.resample_rules,
            "threshold": self.threshold,
            "threshold_correction": self.threshold_correction,
            "stopping_method": self.stopping_method,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

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
        self.stream = StringIO()

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        self.stream.write("\nParsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        self.stream.write("\nValidating the input data")
        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        self.stream.write(
            "\nThe data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
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
        
        # Invert labels if we are in disjunction mode
        if self.model_type == "disjunction":
            self.stream.write("\nInverting labels for disjunction mode")
            self.stream.write(f"\n y before = {y}")
            y = np.logical_not(y).astype(int)
            self.stream.write(f"\n y after  = {y}")

        # Bonferonni correction for the threshold
        if self.threshold_correction == 'bonferroni':
            self.threshold = self.threshold / X.shape[1]
            self.stream.write(f'\nBonferroni correction for the threshold, new threshold = {self.threshold}')

        # Create an empty model
        self.stream.write("\nInitializing empty model")
        if self.model_type == "conjunction":
            self.model_ = ConjunctionModel()
        elif self.model_type == "disjunction":
            self.model_ = DisjunctionModel()
        else:
            raise ValueError("Unsupported model type.")
        self.stream.write("\nTraining start")

        # first column of X: environment
        env_of_remaining_examples = X[:, 0]

        # Extract features only
        X_original_with_env = (
            X.copy()
        )  # useful for prediction for feature importance computation
        X = X[:, 1:]
        remaining_y = y

        all_possible_rules = []
        for feat_id in range(X.shape[1]):
            for threshold in list(set(X[:, feat_id])):
                for kind in ["greater", "less_equal"]:
                    all_possible_rules.append((feat_id, threshold, kind))

        big_pred_matrix = np.zeros((X.shape[0], len(all_possible_rules)), dtype=int)
        for sample_id in range(X.shape[0]):
            x = X[sample_id]
            for rule_id in range(len(all_possible_rules)):
                rule_feat_id, rule_threshold, rule_kind = all_possible_rules[rule_id]
                sample_feat_value = x[rule_feat_id]
                if rule_kind == "greater":
                    pred = 1 if (sample_feat_value > rule_threshold) else 0
                elif rule_kind == "less_equal":
                    pred = 1 if (sample_feat_value <= rule_threshold) else 0
                else:
                    raise ValueError("unexpected rule kind:", rule_kind)
                big_pred_matrix[sample_id, rule_id] = pred

        # Calculate residuals
        residuals = (big_pred_matrix != remaining_y[:, None]).astype(int)
        stopping_criterion = False
        n_rules_with_indep_neg_residuals = [len(all_possible_rules)]

        while not (stopping_criterion):
            n_rules_with_indep_neg_residuals.append(0)
            p_vals_neg_leafs = []
            utilities = []
            scores_of_rules = []
            assert residuals.shape[1] == len(all_possible_rules)
            self.stream.write(f'\nlen(all_possible_rules) = {len(all_possible_rules)}')
            y_e_df = pd.DataFrame(
                {
                    "y": remaining_y,
                    "e": env_of_remaining_examples,
                }
            )
            for i in range(residuals.shape[1]):
                res = residuals[:, i]  # erreurs de la regle
                utility_true_negatives = np.logical_not(
                    np.logical_or(res, remaining_y)
                ).astype(int)
                utility_false_negatives = np.logical_and(res, remaining_y).astype(int)
                utility = sum(utility_true_negatives) - self.p * sum(
                    utility_false_negatives
                )
                rule_feat_id, rule_threshold, rule_kind = all_possible_rules[i]
                utilities.append(utility)
                y_e_df["rule_pred"] = big_pred_matrix[:, i]
                neg_leaf_y_e_df = y_e_df[y_e_df["rule_pred"] == 0]
                if len(neg_leaf_y_e_df) == 0:
                    p_value_neg_leaf = 1
                else:
                    p_value_neg_leaf = stat_test(neg_leaf_y_e_df['y'], neg_leaf_y_e_df['e'])
                p_vals_neg_leafs.append(p_value_neg_leaf)
                n_rules_with_indep_neg_residuals[-1] = n_rules_with_indep_neg_residuals[
                    -1
                ] + int(p_value_neg_leaf > self.threshold)
                score_of_rule = utility if (p_value_neg_leaf > self.threshold) else -np.inf
                scores_of_rules.append(score_of_rule)
                self.stream.write('\nrule : feature {} {:2} {}     p_value_neg_leaf = {:3f} |{:10}|     utility = {:5d}       score = {:5}'.format(rule_feat_id, '>' if rule_kind == 'greater' else '<=', rule_threshold, p_value_neg_leaf, '#'*int(10*p_value_neg_leaf), int(utility), float(score_of_rule)))
            stopping_criterion = False
            if max(scores_of_rules) == -np.inf:
                stopping_criterion = True
                break
            
            p_vals_neg_leafs = np.array(p_vals_neg_leafs)
            best_rule_id = np.array(scores_of_rules).argmax()
            best_rule_score = scores_of_rules[best_rule_id]
            best_rule_feat_id, best_rule_threshold, best_rule_kind = all_possible_rules[
                best_rule_id
            ]
            self.stream.write('\nselected best rule : "feature {} {:2} {}"'.format(best_rule_feat_id, '>' if best_rule_kind == 'greater' else '<=', best_rule_threshold))

            mask = np.zeros(big_pred_matrix.shape, dtype=bool)
            updated_all_possible_rules = all_possible_rules.copy()
            columns_to_keep = np.array(
                [(rule in updated_all_possible_rules) for rule in all_possible_rules]
            )
            predictions_of_selected_rule = big_pred_matrix[:, best_rule_id]
            self.stream.write('\nlen(predictions_of_selected_rule) = {}'.format(len(predictions_of_selected_rule)))
            classified_neg_examples = [
                (r == p == 0)
                for r, p in zip(
                    residuals[:, best_rule_id], predictions_of_selected_rule
                )
            ]
            if sum(classified_neg_examples) == 0:
                self.stream.write('\nthe selected best rule do not classify any negative example: breaking the while loop and ending the model building here')
                break
            samples_to_keep = np.array(predictions_of_selected_rule).astype(bool)
            for i in range(big_pred_matrix.shape[0]):
                if samples_to_keep[i]:
                    mask[i] = columns_to_keep
            new_dimensions = (sum(samples_to_keep), sum(columns_to_keep))
            self.stream.write('\nnew_dimensions = {}'.format(new_dimensions))
            updated_big_pred_matrix = big_pred_matrix[mask].reshape(new_dimensions)
            updated_residuals = residuals[mask].reshape(new_dimensions)
            samples_classed_by_last_rule = np.array([1 - p for p in predictions_of_selected_rule]).astype(bool)
            y_of_classified_examples = remaining_y[samples_classed_by_last_rule]
            env_of_classified_examples = env_of_remaining_examples[samples_classed_by_last_rule]
            remaining_y = remaining_y[samples_to_keep]
            env_of_remaining_examples = env_of_remaining_examples[samples_to_keep]
            big_pred_matrix = updated_big_pred_matrix
            residuals = updated_residuals
            all_possible_rules = updated_all_possible_rules

            global_best_rule_feat_id = best_rule_feat_id + 1
            stump = DecisionStump(
                feature_idx=global_best_rule_feat_id,
                threshold=best_rule_threshold,
                kind=best_rule_kind,
            )

            self.stream.write("\nThe best rule has score {}".format(best_rule_score))
            self._add_attribute_to_model(stump)
            self.stream.write(f'\n model after adding the rule: {self.model_}')
            self.stream.write('\n-------------------')
            self.stream.write(f'\n classified by last rule : {len(y_of_classified_examples):6} examples | {sum(y_of_classified_examples):6} ones | {len(y_of_classified_examples) - sum(y_of_classified_examples):6} zeros')
            y_of_classified_in_E0 = [y for y, e in zip(y_of_classified_examples, env_of_classified_examples) if e == 0]
            y_of_classified_in_E1 = [y for y, e in zip(y_of_classified_examples, env_of_classified_examples) if e == 1]
            self.stream.write(f'\n                      E0 : {len(y_of_classified_in_E0):6} examples | {sum(y_of_classified_in_E0):6} ones | {len(y_of_classified_in_E0) - sum(y_of_classified_in_E0):6} zeros')
            self.stream.write(f'\n                      E1 : {len(y_of_classified_in_E1):6} examples | {sum(y_of_classified_in_E1):6} ones | {len(y_of_classified_in_E1) - sum(y_of_classified_in_E1):6} zeros')
            self.stream.write('\n-------------------')
            self.stream.write(f'\n remaining :               {len(remaining_y):6} examples | {sum(remaining_y):6} ones | {len(remaining_y) - sum(remaining_y):6} zeros')
            y_of_remaining_in_E0 = [y for y, e in zip(remaining_y, env_of_remaining_examples) if e == 0]
            y_of_remaining_in_E1 = [y for y, e in zip(remaining_y, env_of_remaining_examples) if e == 1]
            self.stream.write(f'\n                      E0 : {len(y_of_remaining_in_E0):6} examples | {sum(y_of_remaining_in_E0):6} ones | {len(y_of_remaining_in_E0) - sum(y_of_remaining_in_E0):6} zeros')
            self.stream.write(f'\n                      E1 : {len(y_of_remaining_in_E1):6} examples | {sum(y_of_remaining_in_E1):6} ones | {len(y_of_remaining_in_E1) - sum(y_of_remaining_in_E1):6} zeros')
            self.stream.write('\n-------------------')

            self.stream.write('\nevaluation of stopping conditions : ')
            self.stream.write(f"\nlen(self.model_) >= self.max_rules : {len(self.model_) >= self.max_rules} (len(self.model_) = {len(self.model_)}, self.max_rules = {self.max_rules}")
            self.stream.write(f'\nlen(remaining_y) == 0 : {len(remaining_y) == 0}')
            self.stream.write(f'\nlen(all_possible_rules) == 0 : {len(all_possible_rules) == 0}')
            
            if len(self.model_) >= self.max_rules:
                self.stream.write(f"\nlen(self.model_) >= self.max_rules {len(self.model_)}, {self.max_rules} stopping")
                stopping_criterion = True
            elif len(remaining_y) == 0:
                self.stream.write(f"\nlen(remaining_y) == 0 : {len(remaining_y)} stopping")
                stopping_criterion = True
            elif len(all_possible_rules) == 0:
                self.stream.write(
                   f"\nlen(all_possible_rules) == 0 : {len(all_possible_rules)} stopping")
                stopping_criterion = True
            else:
                if self.stopping_method == "no_more_negatives":
                    self.stream.write(f"\nself.stopping_method == no_more_negatives {(len(remaining_y) == sum(remaining_y))} stopping if True")
                    stopping_criterion = len(remaining_y) == sum(
                        remaining_y
                    )  # only positive examples remaining
                elif self.stopping_method == "independance_y_e":
                    self.stream.write(f'\nindependance_y_e')
                    assert len(remaining_y) == len(env_of_remaining_examples)
                    y_e_df = pd.DataFrame(
                        {"remaining_y": remaining_y, "e": env_of_remaining_examples}
                    )

                    p_value_stopping = stat_test(y_e_df['remaining_y'], y_e_df['e'])
                    stopping_criterion = p_value_stopping > self.threshold
                    self.stream.write(f"\n(p_value_stopping = {p_value_stopping} | self.threshold = {self.threshold}")
                    self.stream.write(f"\n(p_value_stopping > self.threshold) {(p_value_stopping > self.threshold)} (stopping if True)")
                else:
                    raise ValueError(
                        "unexpected stopping_criterion", self.stopping_method
                    )

            self.stream.write(
                "\nDiscarding all examples that the rule classifies as negative"
            )

            iteration_callback(self.model_)

        self.stream.write("\nTraining completed")

        if self.pruning:
            self.stream.write("\nStart pruning model")
            rules_to_keep_mask = [True for _ in self.model_.rules]
            #model_features_idx_to_keep = [rule.feature_idx for rule in self.model_.rules]
            variables_idx = list(range(X_original_with_env.shape[1]))[1:]  # because feature 0 in X_original_with_env is the environment
            conditional_indep_calculation_df = pd.DataFrame(
                X_original_with_env, columns=["e"] + variables_idx
            )
            conditional_indep_calculation_df["y"] = y
            y_position = conditional_indep_calculation_df.columns.get_loc("y")
            e_position = conditional_indep_calculation_df.columns.get_loc("e")
            for i in range(len(rules_to_keep_mask)):
                if sum(rules_to_keep_mask) < 2:
                    break
                rules_to_keep_mask[i] = False
                sub_models_feat_idx = [rule.feature_idx for rule in self.model_.rules if rules_to_keep_mask[i]]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    p_value = ci_tests.ci_test_bin(
                        conditional_indep_calculation_df.values,
                        e_position,
                        y_position,
                        sub_models_feat_idx,
                    )
                self.stream.write(f'\n testing the subset={sub_models_feat_idx}, p_value={round(p_value, 6)}')
                if p_value > self.threshold:
                    # remove rule from model:
                    self.stream.write(f'\n set={sub_models_feat_idx}, p_value={round(p_value, 6)}                         independent ')
                    self.stream.write(f'\n removing this rule from model: {i}')
                    self.stream.write(f'\n len(self.model_.rules) = {len(self.model_.rules)}')
                    break
                else:
                    rules_to_keep_mask[i] = True
                    self.stream.write(f'\n set={sub_models_feat_idx}, p_value={round(p_value, 6)}')
            
            rules_to_keep = [self.model_.rules[i] for i in range(len(self.model_.rules)) if rules_to_keep_mask[i]]
            self.model_.rules = rules_to_keep
            self.stream.write('\n end pruning')
        self.n_rules_with_indep_neg_residuals = n_rules_with_indep_neg_residuals

        self.stream.write("\nCalculating rule importances")
        # Definition: how often each rule outputs a value that causes the value of the model to be final
        final_outcome = {"conjunction": 0, "disjunction": 1}[self.model_type]
        total_outcome = (
            self.model_.predict(X_original_with_env) == final_outcome
        ).sum()  # n times the model outputs the final outcome
        self.rule_importances = np.array(
            [
                (r.classify(X_original_with_env) == final_outcome).sum() / total_outcome
                for r in self.model_.rules
            ]
        )  # contribution of each rule
        self.stream.write(f'\nmodel : {self.model_}')
        self.stream.write(f'\nrule importances: {self.rule_importances}')
        self.stream.write("\nDone.")

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
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
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
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
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
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
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
