import numpy as np
import pandas as pd
import warnings
import graphviz


def find_best_split_con(X, y, is_classification=True):
    best_col_split = None
    best_split_value = None
    best_metric_result = -1
    is_best_split_numeric = True

    for col in range(X.shape[1]):
        conditions, is_numeric = possible_splits(X[:, col])
        for condition in conditions:
            split_metric = metric_split_condition(
                X, y, col, condition, is_numeric, is_classification=is_classification
            )
            if (split_metric < best_metric_result) | (best_metric_result == -1):
                best_col_split = col
                best_metric_result = split_metric
                best_split_value = condition
                is_best_split_numeric = is_numeric

    return best_col_split, best_split_value, best_metric_result, is_best_split_numeric


def gini_inpurity(classes):
    counts = np.unique(classes, return_counts=True)[1]
    n = counts.sum()
    gini = 1
    for count in counts:
        gini -= (count / n) ** 2

    return gini


def sum_squared_residuals(classes):
    mean_test = classes.mean()
    ssr = ((classes - mean_test) ** 2).sum()
    return ssr


def moving_average(x, w):
    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    return np.convolve(x, np.ones(w), "valid") / w


def possible_splits(column):
    try:
        return moving_average(np.sort(column), 2), True
    except TypeError:
        return np.unique(column), False
    else:
        raise Exception("Fail to find splits")


def metric_split_condition(
    X, y, col, condition, is_numeric=True, is_classification=True
):
    if is_numeric:
        indexes_left = np.where(X[:, col] <= condition)
        indexes_right = np.where(X[:, col] > condition)
    else:
        indexes_left = np.where(X[:, col] == condition)
        indexes_right = np.where(X[:, col] != condition)

    if is_classification:
        gini_left = gini_inpurity(y[indexes_left])
        gini_right = gini_inpurity(y[indexes_right])
        result = (
            len(y[indexes_left]) / len(y) * gini_left
            + len(y[indexes_right]) / len(y) * gini_right
        )
        return result
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ssr_left = sum_squared_residuals(y[indexes_left])
            ssr_right = sum_squared_residuals(y[indexes_right])
        return ssr_left + ssr_right


def plot_tree(dtc, shape="box"):
    tree = graphviz.Digraph("tree")
    tree.attr(fixedsize="True", imagescale="True", size="10!")
    tree.node(dtc.root.get_name(), label=dtc.root.to_text(), shape=shape)
    nodes = [dtc.root]
    for depth in range(dtc.max_depth):
        new_nodes = []
        for node in nodes:
            left_node = node.left_child
            right_node = node.right_child
            if left_node:
                # try:
                tree.node(left_node.get_name(), left_node.to_text(), shape=shape)
                tree.edge(node.get_name(), left_node.get_name())
                new_nodes.append(left_node)
                # except Exception as e:
                #    print(e)

            if right_node:
                # try:
                tree.node(right_node.get_name(), right_node.to_text(), shape=shape)
                tree.edge(node.get_name(), right_node.get_name())
                new_nodes.append(right_node)
                # except Exception as e:
                #    print(e)
        nodes = new_nodes

    return tree
