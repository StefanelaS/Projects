# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:47:47 2023

@author: Asus
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import os

np.random.seed(1)

def tki():

    current_directory = os.getcwd()
    data_filename = "tki-resistance.csv"  
    path = os.path.join(current_directory, data_filename)
    data = np.genfromtxt(path, dtype=object, delimiter=',', skip_header=True )  
    unique_classes = np.unique(data[:, -1])  # finding unique classes
    legend = {label: i for i, label in enumerate(unique_classes)}  # creating legend for binary encoding
    data[:,-1] = [legend[label] for label in data[:,-1]]  # encoding the chategorical classes
    learn_data = data[:130, :]  # the first 130 rows of data are used as the training set
    learn = learn_data[:, :-1 ], learn_data[:, -1] # splitting X and y 
    test_data = data[130:, :]  # the rest is used as a testing set
    test = test_data[:, :-1], test_data[:, -1]  # splitting X and y
    return learn, test, legend


def all_columns(X, rand):
    return range(X.shape[1]) # returns all columns indices of X


def random_sqrt_columns(X, rand):
    n_columns = X.shape[1]  # number of features(columns)
    n_candidate_columns = int(np.sqrt(n_columns))  # number of candidate columns is square root of total number of columns
    c = rand.sample(range(n_columns), k=n_candidate_columns) # select random columns without replacement
    return c


class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        if rand is None:
            rand = random.Random(1)
        self.rand = rand  
        self.get_candidate_columns = get_candidate_columns  
        self.min_samples = min_samples

    def build(self, X, y):
        n_samples = X.shape[0]
        y = y.astype('int')
        # Stopping criterion: if there is less than 2 samples or node is pure. If conditions are True, it is a leaf node
        if len(np.unique(y)) == 1:
            return TreeNode(leaf_value = np.bincount(y).argmax())
        if n_samples < self.min_samples: 
            return TreeNode(leaf_value = int(y[0]))
        candidate_columns = self.get_candidate_columns(X, self.rand)
        # Finding the best feature and threshold for a split
        feature, threshold = self.find_best_split(X, y, candidate_columns)
        # If there is no feature or threshold that creates a split with lower gini impurity, it is a leaf node
        if feature is None or threshold is None: 
            return TreeNode(leaf_value = np.bincount(y).argmax())
        # Splitting the parent node into a leaf and right child nodes
        left = X[:, feature] < threshold  # returns np array of True&False, True for values <= threshold
        right = X[:, feature] >= threshold  # returns np array of True&False, True for values > threshold
        yl, yr = y[left], y[right]
        Xl, Xr = X[left], X[right]
        left_child = self.build(Xl, yl)
        right_child = self.build(Xr, yr)
        return TreeNode(feature, threshold, left_child, right_child) # dummy output
      
    
    def find_best_split(self, X, y, candidate_columns):
        best_gini, best_feature, best_value = 1, None, None
        # Looping through all features and possible threshold to find a best split (with lowest gini impurity)
        for feature in candidate_columns:  
            values = X[:, feature]
            sorted_idx = values.argsort()
            values = values[sorted_idx]
            y_sorted = y[sorted_idx]
            for threshold in np.unique(values):  
                index = np.argmax(values == threshold)
                # creating the split
                yl = y_sorted[:index]
                yr = y_sorted[index:]
                if len(yl) == 0 or len(yr) == 0:
                    continue
                # computing gini impurity
                impurity = self.gini_split(y, yl, yr)
                if impurity < best_gini:
                    best_gini = impurity
                    best_feature = feature
                    best_value = threshold
        return best_feature, best_value
    
    def gini_subnode(self, y):
        _, counts = np.unique(y, return_counts=True)  # gets occurancy of each label
        p = counts / np.sum(counts)  # computes proportion of labels
        return 1 - np.sum(p**2)  # returns gini impurity of the subnode
    
    def gini_split(self, y, yl, yr):
        w1 = len(yl) / len(y)
        w2 = len(yr) / len(y)        
        gini = w1 * self.gini_subnode(yl) + w2 * self.gini_subnode(yr)  # weighted gini impurity for a split
        return gini

    
class TreeNode:

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, leaf_value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value
        
    def predict(self, X):
        if self.leaf_value is not None:
            return np.array([self.leaf_value] * len(X))  
        predictions = np.zeros(X.shape[0])
        for i in range(len(X)):
            if X[i, self.feature] < self.threshold:
                predictions[i] = (self.left_child.predict(X[i, :].reshape(1, -1))[0])
            else:
                predictions[i] = (self.right_child.predict(X[i, :].reshape(1, -1))[0])
        return predictions

class RandomForest:

    def __init__(self, rand=None, n=50, NonRandomTrees = False):
        if rand is None:
            rand = random.Random(1)
        self.n = n
        self.rand = rand
        if NonRandomTrees == True:
            self.rftree = Tree(rand=self.rand, get_candidate_columns=all_columns, min_samples=2)  # uses all features
        else:
            self.rftree = Tree(rand=self.rand, get_candidate_columns=random_sqrt_columns, min_samples=2)  # uses sqrt random features

    def build(self, X, y):
        models = []
        X_oob = []  # place for storing out-of-box X samples in each bootstrap
        y_oob = []  # place for storing out-of-box y samples in each bootstrap
        n_samples = X.shape[0]
        for i in range(self.n):
           bootstrap_idx = self.rand.choices(range(n_samples), k=n_samples)
           bootstrap_X = X[bootstrap_idx, :]
           bootstrap_y = y[bootstrap_idx]
           oob_idx = list(set(range(n_samples)) - set(bootstrap_idx))
           X_oob.append(X[oob_idx, :])
           y_oob.append(y[oob_idx])
           model = self.rftree.build(bootstrap_X, bootstrap_y)
           models.append(model)
        oob_samples = (X_oob, y_oob)
        return RFModel(models, oob_samples)


class RFModel:
    
    def __init__(self, models, oob_samples):
        self.models = models
        self.oob_samples = oob_samples

    def predict(self, X):
        n_samples = X.shape[0]
        all_predictions = np.zeros((n_samples, len(self.models)))  # creating empty np array of size (number of samples, number of models)
        for i, model in enumerate(self.models):
            all_predictions[:, i] = model.predict(X) # storing predictions from each model
        class_occ = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=2), axis=1, arr=all_predictions)
        predictions = np.argmax(class_occ, axis=1)
        return predictions
    
    def importance(self):
        X_oob, y_oob = self.oob_samples
        n_features = X_oob[0].shape[1]
        imps = np.zeros(n_features)
        for i, tree in enumerate(self.models):
            model_imps = np.zeros(n_features)
            y_pred = tree.predict(X_oob[i])
            baseline_correct = np.sum(y_pred == y_oob[i])
            baseline_mc_rate = 1 - (baseline_correct / len(y_pred))
            for j in range(n_features):
                original_column = X_oob[i][:,j].copy()
                random.Random(j).shuffle(X_oob[i][:,j])
                perm_pred = tree.predict(X_oob[i])
                perm_correct = np.sum(perm_pred == y_oob[i])
                perm_mc_rate = 1 - (perm_correct / len(perm_pred))
                model_imps [j] = 100*(perm_mc_rate-baseline_mc_rate)/baseline_mc_rate
                X_oob[i][:, j] = original_column
            imps += model_imps
        imps /= len(self.models)
        return imps
            
            
def hw_tree_full(learn, test):
    X_train, y_train = learn
    X_test, y_test = test
    # building a simple decision tree
    tree = Tree(min_samples=2)
    model = tree.build(X_train, y_train)
    
    # performance on training set
    y_pred_train = model.predict(X_train)
    correct_train = np.sum(y_train == y_pred_train)
    train_mc_rate = 1 - (correct_train / len(y_train))
    std_train = standard_deviation(y_train, y_pred_train, 1000)
    
    # performance on test set
    y_pred_test = model.predict(X_test)
    correct_test = np.sum(y_test == y_pred_test)
    test_mc_rate = 1 - (correct_test / len(y_test))
    std_test = standard_deviation(y_test, y_pred_test, 1000)
    
    return (train_mc_rate, std_train), (test_mc_rate, std_test)

 
def hw_randomforests (learn, test):
    X_train, y_train = learn
    X_test, y_test = test
    rf = RandomForest(n=100)
    rf_model = rf.build(X_train, y_train)
    y_pred_test = rf_model.predict(X_test)
    correct_test = np.sum(y_test == y_pred_test)
    test_mc_rate = 1 - (correct_test / len(y_test))

    # performance on training set
    y_pred_train = rf_model.predict(X_train)
    correct_train = np.sum(y_train == y_pred_train)
    train_mc_rate = 1 - (correct_train / len(y_train))
    std_train = standard_deviation(y_train, y_pred_train, 1000)
    
    # performance on test set
    y_pred_test = rf_model.predict(X_test)
    correct_test = np.sum(y_test == y_pred_test)
    test_mc_rate = 1 - (correct_test / len(y_test))
    std_test = standard_deviation(y_test, y_pred_test, 1000)
    
    return (train_mc_rate,std_train), (test_mc_rate, std_test)


def standard_deviation(y_true, y_pred, n):
    n_samples = len(y_pred)
    mc_rates = []
    for i in range(n):
       bootstrap_idx = random.Random(i).choices(range(n_samples), k=n_samples)
       correct = np.sum(y_true[bootstrap_idx] == y_pred[bootstrap_idx])
       mc_rate = 1 - (correct / n_samples)
       mc_rates.append(mc_rate)
    std = np.std(mc_rates)
    return std


def plot_misclassification_versus_n(learn, test):
    X_train, y_train = learn
    X_test, y_test = test
    n = [i for i in range(1, 101, 1)]
    mc_rates_test = []
    mc_rates_train = []

    for i in n:
        rf = RandomForest(n=i)
        rf_model = rf.build(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        correct = np.sum(y_test == y_pred)
        mc_rate = (1 - (correct / len(y_test)))*100
        mc_rates_test.append(mc_rate)
        y_pred_train = rf_model.predict(X_train)
        correct_train = np.sum(y_train == y_pred_train)
        mc_rate_train = (1 - (correct_train / len(y_train)))*100
        mc_rates_train.append(mc_rate_train)
        print(i)

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(n, mc_rates_train, '-', label = 'Train Set', color ='b')
    plt.plot(n, mc_rates_test, label = 'Test Set', color ='r')
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, max(mc_rates_test), 5))
    plt.legend()
    plt.xlabel('Number of Trees')
    plt.ylabel('Misclassification Rate (%)')


def feature_importances(learn):
    X_train, y_train = learn
    start = time.time()
    rf = RandomForest(n=100)
    rf_model = rf.build(X_train, y_train)
    imps_rf = rf_model.importance() 
    end = time.time()
    print('RF feature importances', end - start)

    trees = RandomForest(n=100, NonRandomTrees = True)
    trees_model = trees.build(X_train, y_train)
    imps_nr = trees_model.importance() 
    
    plt.style.use('ggplot')
    
    plt.figure(figsize=(7,4))
    plt.plot(range(0,101), imps_rf[:101], 'o', label = 'Random Forest with 100 Trees')
    plt.plot(range(0,101), imps_nr[:101], '*', label = '100 Non-Random Trees')
    plt.xticks(np.arange(0,101,5), rotation = 45)
    plt.title('a)')
    plt.xlabel('Feature')
    plt.ylabel('Misclassification Rate Increase (%)')
    plt.legend()
    
    plt.figure(figsize=(7,4))
    plt.plot(range(100, X_train.shape[1]), imps_rf[100:X_train.shape[1]], 'o', label = 'Random Forest with 100 Trees')
    plt.plot(range(100, X_train.shape[1]), imps_nr[100:X_train.shape[1]], '*', label = '100 Non-Random Trees')
    plt.xticks(np.arange(100,X_train.shape[1],5), rotation = 45)
    plt.title('b)')
    plt.xlabel('Feature')
    plt.ylabel('Misclassification Rate Increase (%)')
    
    
    # Creating a dataframe with feature importance for both Random Forest with 100 Trees and Non-Random 100 Trees
    feature_importances = pd.DataFrame({'Feature': range(X_train.shape[1]), 'Random Forest': imps_rf, 'Non-Random Trees': imps_nr})
    # Sorting the dataframe by Random Forest feature importance
    importances_sorted = feature_importances.sort_values('Random Forest', ascending=False)
    return importances_sorted

if __name__ == "__main__":
    learn, test, legend = tki()
    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
    importances = feature_importances(learn)
    #plot_misclassification_versus_n(learn, test)