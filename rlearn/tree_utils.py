import numpy as np
import pandas as pd
import warnings
import graphviz

def find_best_split(data, target_col, is_classification=True, parent_score=10e10):
    '''
    
    return: best_col_split, best_split_value, best_metric_result, best_is_numeric
    '''
    best_col_split = None
    best_split_value = None
    best_metric_result = parent_score
    best_is_numeric = True
    for col in data.drop(target_col, axis=1).columns:
        is_numeric = is_col_numeric(data, col)
        possible_conditions = find_numeric_condition(data, col) if is_numeric else find_categorical_codition(data, col)
        
        for condition in possible_conditions:
            new_metric_result = metric_split_condition(data, col, condition, target_col, is_numeric, is_classification)
            
            if new_metric_result < best_metric_result:
                best_col_split = col
                best_metric_result = new_metric_result
                best_split_value = condition
                best_is_numeric = is_numeric
    return best_col_split, best_split_value, best_metric_result, best_is_numeric

def find_numeric_condition(data, col):
    return data[col].sort_values().rolling(2).mean().dropna().to_numpy()

def find_categorical_codition(data, col):
    return data[col].unique().to_numpy()


def metric_split_condition(data, col, split_condition, target_col, is_numeric, is_classification):
    condition_mask = data[col] <= split_condition if is_numeric else data[col] == split_condition
    labels_left = data[condition_mask][target_col].to_numpy()
    labels_right = data[~condition_mask][target_col].to_numpy()
    if is_classification:
        gini_left = gini_inpurity(labels_left)
        gini_right = gini_inpurity(labels_right)
        result = len(labels_left)/len(data) * gini_left + len(labels_right)/len(data) * gini_right
        return result
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ssr_left = sum_squared_residuals(labels_left)
            ssr_right = sum_squared_residuals(labels_right)
        return ssr_left + ssr_right
    
def gini_inpurity(classes):
    counts = np.unique(classes, return_counts=True)[1]
    n = counts.sum()
    gini = 1
    for count in counts:
        gini -= (count/n)**2

    return gini

def sum_squared_residuals(classes):
    mean_test = classes.mean()
    ssr = ((classes-mean_test)**2).sum()
    return ssr

def is_col_numeric(data, col):
    return pd.to_numeric(data[col], errors='coerce').notnull().all()

def plot_tree(dtc, shape='box'):
    tree = graphviz.Digraph('tree')
    tree.attr(fixedsize = 'True', imagescale='True', size='10!')
    tree.node(dtc.root.get_name(), label=dtc.root.to_text(), shape=shape)
    nodes = [dtc.root]
    for depth in range(dtc.max_depth):
        new_nodes = []
        for node in nodes:
            left_node = node.left_child
            right_node = node.right_child
            if left_node:
                try:
                    tree.node(left_node.get_name(), left_node.to_text(), shape=shape)
                    tree.edge(node.get_name(), left_node.get_name())
                    new_nodes.append(left_node)
                except:
                    pass

            if right_node:
                try:
                    tree.node(right_node.get_name(), right_node.to_text(), shape=shape)
                    tree.edge(node.get_name(), right_node.get_name())
                    new_nodes.append(right_node)
                except:
                    pass
        nodes = new_nodes
            
    return tree