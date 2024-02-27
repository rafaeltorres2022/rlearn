import numpy as np
import pandas as pd
from rlearn.tree_utils import *
from random import sample
from rlearn.metrics import MeanSquaredError

class Node:
    def __init__(self, col_split = None, value_to_split = None, evaluation_metric = 1, is_condition_numeric=True, is_leaf = False, data=None) -> None:
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
        condition = f'{self.col_split}<={self.value_to_split}' if self.is_condition_numeric else f'{self.col_split}=={self.value_to_split}'
        metric = f'Gini: {self.evaluation_metric}' if self.is_classification else f'Sum of Squared Residuals: {self.evaluation_metric}'
        return f'{condition}\n{metric}\nLeaf: {self.is_leaf}\nN. Samples: {self.n_samples}\nOutput: {self.output}'
    
    def to_text(self) -> str:
        condition_value = f'{"{:.3f}".format(float(self.value_to_split))}' if self.is_condition_numeric else self.value_to_split
        condition = f'{self.col_split}<={condition_value}' if self.is_condition_numeric else f'{self.col_split}=={condition_value}'
        metric = f'Gini: {"{:.3f}".format(self.evaluation_metric)}' if self.is_classification else f'SSR: {"{:.3f}".format(self.evaluation_metric)}'
        leaf = f'\nLeaf: {self.is_leaf}' if self.is_leaf else ''
        text = f'{condition}\n{metric}{leaf}\nN. Samples: {self.n_samples}'
        if self.is_leaf:
            text+=f'\nOutput: {"{:.3f}".format(self.output)}'
        return text

    def get_name(self) -> str:
        return str(self.__hash__)

    def set_split_info(self, best_col_split, best_split_value_to_split, best_evaluation_metric, is_condition_numeric):
        self.col_split = best_col_split
        self.value_to_split = best_split_value_to_split
        self.evaluation_metric = best_evaluation_metric
        self.is_condition_numeric = is_condition_numeric

    def choose_output(self, target_col):
        if self.is_classification:
            self.output = self.data[target_col].mode()[0]
        else:
            self.output = self.data[target_col].mean()

    def set_data(self, data):
        self.data = data
        self.n_samples = len(data)

    def define_as_leaf(self, target_col):
        self.is_leaf = True
        self.choose_output(target_col)

class DecisionTreeClassifier:
    def __init__(self, max_depth = 2, min_samples_split = 2) -> None:
        self.root = Node()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.is_classifier = True

    def fit(self, X_train : pd.DataFrame, y_train : pd.Series):
        data = pd.concat([X_train, y_train], axis=1)
        target_col = y_train.name
        self.root.set_split_info(*find_best_split(data, target_col, self.is_classifier))
        self.root.set_data(data)
        self.root.is_classification = self.is_classifier
        parent_nodes = [self.root]
        total_time = 0
        total_time_r = 0
        for depth in range(self.max_depth):
            new_parent_nodes = []
            for parent_node in parent_nodes:
                if parent_node.is_leaf: continue
                if parent_node.is_condition_numeric:
                    data_left = parent_node.data[parent_node.data[parent_node.col_split] <= parent_node.value_to_split]
                    data_right = parent_node.data[~(parent_node.data[parent_node.col_split] <= parent_node.value_to_split)]
                else:
                    data_left = parent_node.data[parent_node.data[parent_node.col_split] == parent_node.value_to_split]
                    data_right = parent_node.data[~(parent_node.data[parent_node.col_split] == parent_node.value_to_split)]
                
                parent_node.left_child = self.evaluate_new_node(data_left, target_col, parent_node, depth)
                parent_node.right_child = self.evaluate_new_node(data_right, target_col, parent_node, depth)
                
                new_parent_nodes.append(parent_node.left_child)
                new_parent_nodes.append(parent_node.right_child)
                parent_node.data = None
            parent_nodes = new_parent_nodes

    def evaluate_new_node(self, data, target_col, parent_node, depth):
        node = Node(*find_best_split(data, target_col, self.is_classifier))
        node.set_data(data)
        node.is_classification = self.is_classifier
        node.evaluation_metric = self.split_metric(data, target_col)
        if ((depth+1 == self.max_depth) | (node.evaluation_metric == 0) | (len(node.data) < self.min_samples_split)):
            node.define_as_leaf(target_col)
        return node

    def split_metric(self, data, target_col):
        return gini_inpurity(data[target_col])

    def predict(self, data):

        return data.apply(lambda row: self.predict_row(row), axis=1)

    def predict_row(self, row):
        next_node = self.root
        for depth in range(self.max_depth):
            next_node = next_node.left_child if (row[next_node.col_split] <= next_node.value_to_split) else next_node.right_child
            if next_node.is_leaf:
                return next_node.output
            
class DecisionTreeRegressor(DecisionTreeClassifier):

    def __init__(self, max_depth=2, min_samples_split=2) -> None:
        super().__init__(max_depth, min_samples_split)
        self.is_classifier = False

    def split_metric(self, data, target_col):
        return sum_squared_residuals(data[target_col])
    
class RandomForestRegressor:

    def __init__(self, max_depth=2, min_samples_split=2, n_estimators = 200, bootstrap_size = 500, n_of_features = 5) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.estimators = []
        self.bootstrap_size = bootstrap_size
        self.n_of_features = n_of_features
        self.target_col = ''
        self.history = {
            'train':[],
            'test':[]
            }
        self.estimator = DecisionTreeRegressor
        self.mse = MeanSquaredError()

    def fit(self, X_train, y_train, X_test, y_test, verbose=5):
        self.target_col = y_train.name
        feature_index = [_ for _ in range(len(X_train.columns))]

        for count in range(self.n_estimators):
            bootstrap = X_train.sample(self.bootstrap_size, replace=True).loc[:, ]
            bootstrap = bootstrap.iloc[:, sample(feature_index, self.n_of_features)]
            labels = y_train.loc[bootstrap.index]



            reg_tree = self.estimator(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            reg_tree.fit(bootstrap, labels)
            self.estimators.append(reg_tree)

            self.save_to_history(X_train, y_train, X_test, y_test)

            if (count%verbose == 0) | (count+1 == self.n_estimators):
                print(f'Train Loss: {self.history["train"][-1]}\tTest Loss: {self.history["test"][-1]}')

    def save_to_history(self, X_train, y_train, X_test, y_test):
        self.history['train'].append(self.mse.loss(y_train, self.predict(X_train)))
        self.history['test'].append(self.mse.loss(y_test, self.predict(X_test)))


    def predict(self, data):
        predictions = pd.DataFrame([])
        for tree in self.estimators:
            predictions = pd.concat([predictions, tree.predict(data)], axis=1)
        return predictions.mean(axis=1)
    
class RandomForestClassifier(RandomForestRegressor):

    def __init__(self, max_depth=2, min_samples_split=2, n_estimators=200, bootstrap_size=500, n_of_features=5) -> None:
        super().__init__(max_depth, min_samples_split, n_estimators, bootstrap_size, n_of_features)
        self.estimator = DecisionTreeClassifier

    def save_to_history(self, X_train, y_train, X_test, y_test):
        self.history['train'].append((y_train.values == self.predict(X_train).values).sum() / len(X_train))
        self.history['test'].append((y_test.values == self.predict(X_test).values).sum() / len(X_test))

    def predict(self, data):
        predictions = pd.DataFrame([])
        for tree in self.estimators:
            predictions = pd.concat([predictions, tree.predict(data)], axis=1)
        return predictions.mode(axis=1).iloc[:,0]
    
class GradientBoostRegressor:

    def __init__(self, learning_rate = 0.1, n_estimators = 100, frac_of_samples = 1, max_features = -1, min_samples_split = 2, max_depth = 2) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.frac_of_samples = frac_of_samples
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.first_prediction = 0
        self.estimators = []
        self.mse = MeanSquaredError()
        self.history = {
            'train':[],
            'test':[]
        }
        
    def fit(self, X_train, y_train, X_test, y_test, verbose=10):
        self.first_prediction = y_train.mean()
        n = len(X_train)
        n_cols = len(X_train.columns)
        if self.max_features == -1: self.max_features = n_cols

        for n_estimator in range(self.n_estimators):
            residuals = y_train - self.predict(X_train)
            residuals.rename('Residuals', inplace=True)
            new_estimator = DecisionTreeRegressor(self.max_depth, self.min_samples_split)
            subsample = X_train.sample(
                int(n * self.frac_of_samples)).iloc[
                    :, sample(range(n_cols),
                        k=self.max_features)]
            new_estimator.fit(subsample, residuals.loc[subsample.index])
            self.estimators.append(new_estimator)

            self.save_history(X_train, y_train, X_test, y_test)
            
            if (n_estimator % verbose == 0) | (n_estimator+1 == self.n_estimators):
                self.print_loss(n_estimator)

    def predict(self, data):
        pred = self.first_prediction
        for estimator in self.estimators:
            pred += estimator.predict(data) * self.learning_rate

        return pred
    
    def save_history(self, X_train, y_train, X_test, y_test):
        self.history['train'].append(
            self.mse.loss(y_train, self.predict(X_train))
        )
        self.history['test'].append(
            self.mse.loss(y_test, self.predict(X_test))
        )

    def print_loss(self, n_estimator):
        print(f'Estimators: {n_estimator}\tTrain Loss: {self.history["train"][-1]}\tValidation Loss: {self.history["test"][-1]}')