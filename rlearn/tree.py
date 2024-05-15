import numpy as np
import pandas as pd
from rlearn.tree_utils import *
from random import sample
from rlearn.metrics import MeanSquaredError
from scipy.special import softmax
from scipy.stats import mode


class Node:
    def __init__(
        self,
        col_split=None,
        value_to_split=None,
        evaluation_metric=1,
        is_condition_numeric=True,
        is_leaf=False,
        data=None,
    ) -> None:
        self.col_split = col_split
        self.value_to_split = value_to_split
        self.evaluation_metric = evaluation_metric
        self.is_leaf = is_leaf
        self.left_child = None
        self.right_child = None
        self.data = data
        self.output = None
        self.is_condition_numeric = is_condition_numeric
        self.n_samples = 0
        self.is_classification = True

    def __repr__(self) -> str:
        condition = (
            f"{self.col_split}<={self.value_to_split}"
            if self.is_condition_numeric
            else f"{self.col_split}=={self.value_to_split}"
        )
        metric = (
            f"Gini: {self.evaluation_metric}"
            if self.is_classification
            else f"Sum of Squared Residuals: {self.evaluation_metric}"
        )
        return f"{condition}\n{metric}\nLeaf: {self.is_leaf}\nN. Samples: {self.n_samples}\nOutput: {self.output}"

    def to_text(self) -> str:
        condition_value = (
            f'{"{:.3f}".format(float(self.value_to_split))}'
            if self.is_condition_numeric
            else self.value_to_split
        )
        condition = (
            f"{self.col_split}<={condition_value}"
            if self.is_condition_numeric
            else f"{self.col_split}=={condition_value}"
        )
        metric = (
            f'Gini: {"{:.3f}".format(self.evaluation_metric)}'
            if self.is_classification
            else f'SSR: {"{:.3f}".format(self.evaluation_metric)}'
        )
        text = f"{condition}\n{metric}\nLeaf: {self.is_leaf}"
        if self.is_leaf:
            text += f"\nN. Samples: {self.n_samples}\nOutput: {self.output}"
        return text

    def get_name(self) -> str:
        return str(self.__hash__)

    def set_split_info(
        self,
        best_col_split,
        best_split_value_to_split,
        best_evaluation_metric,
        is_condition_numeric,
    ):
        self.col_split = best_col_split
        self.value_to_split = best_split_value_to_split
        self.evaluation_metric = best_evaluation_metric
        self.is_condition_numeric = is_condition_numeric

    def choose_output(self, data):
        if self.is_classification:
            values, counts = np.unique(data, return_counts=True)
            self.output = values[np.argmax(counts)]
        else:
            self.output = np.mean(data)

    def set_data(self, data):
        self.data = data
        self.n_samples = len(data)

    def define_as_leaf(self, data, is_classification=True):
        self.is_leaf = True
        self.is_classification = is_classification
        self.set_data(data)
        self.choose_output(data)


class DecisionTreeClassifier:
    def __init__(self, max_depth=2, min_samples_split=2) -> None:
        self.root = Node()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.is_classifier = True
        self.longest_string_result_size = 0

    def fit(self, X_train, y_train):
        try:
            self.longest_string_result_size = len(max(y_train, key=len))
        except TypeError:
            pass

        self.root = self.build_node(X_train, y_train, depth=0)

    def build_node(self, X, y, depth):
        new_node = Node()
        if self.is_classifier:
            node_metric = gini_inpurity(y)
        else:
            new_node.is_classification = False
            node_metric = sum_squared_residuals(y)

        best_col_split, best_split_value, best_metric_result, is_best_split_numeric = (
            find_best_split_con(X, y, self.is_classifier)
        )
        new_node.set_split_info(
            best_col_split, best_split_value, node_metric, is_best_split_numeric
        )

        if is_best_split_numeric:
            left_mask = np.where(X[:, best_col_split] <= best_split_value)
            right_mask = np.where(X[:, best_col_split] > best_split_value)
        else:
            left_mask = np.where(X[:, best_col_split] == best_split_value)
            right_mask = np.where(X[:, best_col_split] != best_split_value)

        # check pruning conditions
        if (
            (depth == self.max_depth)
            | (len(y) < self.min_samples_split)
            | (node_metric == 0.00)
        ):
            new_node.define_as_leaf(y, self.is_classifier)
            return new_node
        # recursive call
        new_node.left_child = self.build_node(
            X[left_mask],
            y[left_mask],
            depth + 1,
        )
        new_node.right_child = self.build_node(
            X[right_mask],
            y[right_mask],
            depth + 1,
        )
        return new_node

    def predict(self, X):
        return np.apply_along_axis(self.predict_row, 1, X)

    def predict_row(self, row):
        next_node = self.root
        for depth in range(self.max_depth):
            if next_node.is_condition_numeric:
                next_node = (
                    next_node.left_child
                    if (row[next_node.col_split] <= next_node.value_to_split)
                    else next_node.right_child
                )
            else:
                next_node = (
                    next_node.left_child
                    if (row[next_node.col_split] == next_node.value_to_split)
                    else next_node.right_child
                )
            if next_node.is_leaf:
                try:
                    return np.array([next_node.output]).astype(float)
                except ValueError:
                    return np.array([next_node.output]).astype(
                        f"<U{self.longest_string_result_size}"
                    )


class DecisionTreeRegressor(DecisionTreeClassifier):
    def __init__(self, max_depth=2, min_samples_split=2) -> None:
        super().__init__(max_depth, min_samples_split)
        self.is_classifier = False

    def split_metric(self, data, target_col):
        return sum_squared_residuals(data[target_col])


class RandomForestRegressor:
    def __init__(
        self,
        max_depth=2,
        min_samples_split=2,
        n_estimators=200,
        bootstrap_size=-1,
        n_of_features=-1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.estimators = []
        self.bootstrap_size = bootstrap_size
        self.n_of_features = n_of_features
        self.history = {"train": [], "test": []}
        self.estimator = DecisionTreeRegressor
        self.mse = MeanSquaredError()
        self.precs = None

    def fit(self, X_train, y_train, X_test, y_test, verbose=5):
        if self.bootstrap_size == -1:
            self.bootstrap_size = len(X_train)
        if self.n_of_features == -1:
            self.n_of_features = X_train.shape[1]
        for count in range(self.n_estimators):
            bootstrap_x = np.random.choice(len(X_train), size=self.bootstrap_size)
            bootstrap_features = np.random.choice(
                X_train.shape[1], size=self.n_of_features, replace=False
            )

            reg_tree = self.estimator(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            reg_tree.fit(
                X_train[
                    np.ix_(bootstrap_x, np.sort(bootstrap_features))
                ],  # bootstrap_x],
                y_train[bootstrap_x],
            )
            self.estimators.append(reg_tree)

            self.save_to_history(X_train, y_train, X_test, y_test)

            self.print_train_log(count, verbose)

    def print_train_log(self, count, verbose):
        if (count % verbose == 0) | (count + 1 == self.n_estimators):
            print(
                f'Train Loss: {self.history["train"][-1]}\tTest Loss: {self.history["test"][-1]}'
            )

    def save_to_history(self, X_train, y_train, X_test, y_test):
        self.history["train"].append(self.mse.loss(y_train, self.predict(X_train)))
        self.history["test"].append(self.mse.loss(y_test, self.predict(X_test)))

    def predict(self, data):
        predictions = self.estimators[0].predict(data)
        for tree in self.estimators[1:]:
            predictions = np.c_[predictions, tree.predict(data)]
        return np.mean(predictions, axis=1)


class RandomForestClassifier(RandomForestRegressor):
    def __init__(
        self,
        max_depth=2,
        min_samples_split=2,
        n_estimators=200,
        bootstrap_size=-1,
        n_of_features=-1,
    ) -> None:
        super().__init__(
            max_depth, min_samples_split, n_estimators, bootstrap_size, n_of_features
        )
        self.estimator = DecisionTreeClassifier
        self.precs = None

    def save_to_history(self, X_train, y_train, X_test, y_test):
        self.history["train"].append(
            (y_train == self.predict(X_train)).sum() / len(X_train)
        )
        self.history["test"].append(
            (y_test == self.predict(X_test)).sum() / len(X_test)
        )

    def print_train_log(self, count, verbose):
        if (count % verbose == 0) | (count + 1 == self.n_estimators):
            print(
                f'Accuracy: {self.history["train"][-1]}\tAccuracy: {self.history["test"][-1]}'
            )

    def predict(self, data):
        predictions = self.estimators[0].predict(data)
        for tree in self.estimators[1:]:
            predictions = np.c_[predictions, tree.predict(data)]
        return mode(predictions, axis=1)[0]


class GradientBoostRegressor:
    def __init__(
        self,
        learning_rate=0.1,
        n_estimators=100,
        frac_of_samples=1,
        max_features=-1,
        min_samples_split=2,
        max_depth=2,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.frac_of_samples = frac_of_samples
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.first_prediction = 0
        self.estimators = []
        self.loss_function = MeanSquaredError()
        self.history = {"train": [], "test": []}
        self.best_n_estimators = 0

    def fit(self, X_train, y_train, X_test, y_test, verbose=10):
        self.first_prediction = y_train.mean()
        n = len(X_train)
        n_cols = len(X_train.columns)
        if self.max_features == -1:
            self.max_features = n_cols

        for n_estimator in range(self.n_estimators):
            residuals = y_train - self.predict(X_train, False)
            residuals.rename("Residuals", inplace=True)
            new_estimator = DecisionTreeRegressor(
                self.max_depth, self.min_samples_split
            )
            subsample = X_train.sample(int(n * self.frac_of_samples)).iloc[
                :, sample(range(n_cols), k=self.max_features)
            ]
            new_estimator.fit(subsample, residuals.loc[subsample.index])
            self.estimators.append(new_estimator)

            self.save_history(X_train, y_train, X_test, y_test)

            if (n_estimator % verbose == 0) | (n_estimator + 1 == self.n_estimators):
                self.print_loss(n_estimator)

            self.check_early_stop(n_estimator)

    def check_early_stop(self, epoch):
        if self.is_new_score_better():
            self.best_n_estimators = epoch

    def predict(self, data, return_best=True):
        pred = self.first_prediction
        for n_estimator, estimator in enumerate(self.estimators):
            if return_best & (n_estimator > self.best_n_estimators):
                break
            pred += estimator.predict(data) * self.learning_rate

        return pred

    def is_new_score_better(self):
        try:
            return self.history["test"][-1] < np.min(self.history["test"][:-1])
        except:
            return True

    def save_history(self, X_train, y_train, X_test, y_test):
        self.history["train"].append(
            self.loss_function.loss(y_train, self.predict(X_train, False))
        )
        self.history["test"].append(
            self.loss_function.loss(y_test, self.predict(X_test, False))
        )

    def print_loss(self, n_estimator):
        print(
            f'Estimators: {n_estimator}\tTrain Loss: {self.history["train"][-1]}\tValidation Loss: {self.history["test"][-1]}'
        )


class GradientBoostClassifier(GradientBoostRegressor):
    def __init__(
        self,
        learning_rate=0.001,
        n_estimators=100,
        frac_of_samples=1,
        max_features=-1,
        min_samples_split=2,
        max_depth=2,
    ) -> None:
        super().__init__(
            learning_rate,
            n_estimators,
            frac_of_samples,
            max_features,
            min_samples_split,
            max_depth,
        )
        self.initial_log_odds = None
        self.classes = None

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=10):
        self.classes = y_train.unique()
        n = len(X_train)
        n_cols = len(X_train.columns)
        if self.max_features == -1:
            self.max_features = n_cols
        oh_y = pd.DataFrame(
            [y_train.iloc[row] == self.classes for row in range(len(y_train))],
            columns=self.classes,
            index=X_train.index,
        ).astype("uint8")
        self.initial_log_odds = np.apply_along_axis(self.log_odds, 0, oh_y)
        initial_prob = softmax(self.initial_log_odds)
        # self.initial_log_odds = pd.DataFrame([initial_log_odds for row in range(len(oh_y))], columns=classes)
        prev_prob = pd.DataFrame(
            [initial_prob for row in range(len(oh_y))],
            columns=self.classes,
            index=X_train.index,
        )
        prev_residuals = oh_y - initial_prob
        for epoch in range(self.n_estimators):
            subsample = X_train.sample(int(n * self.frac_of_samples)).iloc[
                :, sample(range(n_cols), k=self.max_features)
            ]

            new_estimators = self.fit_new_estimators(
                prev_residuals.loc[subsample.index], subsample
            )
            self.transform_leaf_output_into_log_odds(new_estimators, prev_prob, X_train)
            self.estimators.append(new_estimators)

            new_probs = self.predict(X_train, True, False)

            prev_residuals = oh_y - new_probs
            prev_prob = new_probs

            self.save_history(X_train, y_train, X_test, y_test)
            if (epoch % verbose == 0) | (epoch + 1 == self.n_estimators):
                self.print_loss(epoch)

            self.check_early_stop(epoch)

    def predict(self, data, return_raw=False, return_best=True):
        pred = pd.DataFrame(
            [self.initial_log_odds for row in range(len(data))],
            columns=self.classes,
            index=data.index,
        )
        for epoch, epoch_estimators in enumerate(self.estimators):
            if return_best & (epoch > self.best_n_estimators):
                break
            for col_n, estimator in enumerate(epoch_estimators):
                pred.iloc[:, col_n] += estimator.predict(data) * self.learning_rate
        probabilities = pd.DataFrame(
            softmax(pred), columns=self.classes, index=data.index
        )
        if return_raw:
            return probabilities
        else:
            return probabilities.idxmax(axis=1)

    def save_history(self, X_train, y_train, X_test, y_test):
        self.history["train"].append(
            self.accuracy(y_train, self.predict(X_train, return_best=False))
        )
        self.history["test"].append(
            self.accuracy(y_test, self.predict(X_test, return_best=False))
        )

    def is_new_score_better(self):
        try:
            return self.history["test"][-1] > np.max(self.history["test"][:-1])
        except:
            return True

    def log_odds_of_prob(self, leaf, prev_prob):
        sum_of_residuals = leaf.data["Residuals"].sum()
        prev_data = prev_prob.loc[leaf.data.index]
        sum_of_prev_prob = (prev_data * (1 - prev_data)).sum()
        return sum_of_residuals / (sum_of_prev_prob + 10e-8)

    def transform_leaf_output_into_log_odds(self, estimators, prev_prob, X_train):
        for col_n, estimator in enumerate(estimators):
            leafs = estimator.predict(X_train, return_node=True)
            pb = prev_prob.iloc[:, col_n]
            for leaf in leafs.unique():
                new_output = self.log_odds_of_prob(leaf, pb)
                leaf.output = new_output

    def fit_new_estimators(self, prev_residuals, X_train):
        new_estimators = []
        for col in prev_residuals.columns:
            new_estimator = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            new_estimator.fit(
                X_train,
                pd.Series(prev_residuals[col], name="Residuals", index=X_train.index),
            )
            new_estimators.append(new_estimator)
        return new_estimators

    def log_odds(self, column):
        n = len(column)
        count_target = len(column[column == 1])
        return np.log((count_target / n) / (1 - (count_target / n)))

    def accuracy(self, y_true, pred):
        return len(y_true[y_true == pred]) / len((y_true))
