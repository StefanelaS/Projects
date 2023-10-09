# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 00:05:58 2023

@author: Asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

MBOG_TRAIN = 500

def load_data():
    
    df = pd.read_csv('dataset.csv', sep=";")    
    
    # get unique categories and their counts
    print('Entries in each class:')
    categories, counts = np.unique(df['ShotType'], return_counts=True)
    for cat, count in zip(categories, counts):
        print(f'{cat}: {count}')
        
    # encoding of the target variable
    classes_legend = {'above head': 0, 'hook shot': 1, 'layup': 2, 'tip-in': 3, 'dunk': 4, 'other': 5}
    y = df['ShotType'].replace(classes_legend)
    
    # one hot encoding of the cathegorical features
    competition_dummies = pd.get_dummies(df['Competition'], drop_first=True)
    player_dummies = pd.get_dummies(df['PlayerType'], drop_first=True)
    movement_dummies = pd.get_dummies(df['Movement'], drop_first=True)
    df = pd.concat([df, competition_dummies, player_dummies, movement_dummies], axis=1)
    X = df.drop(['Competition', 'PlayerType', 'Movement', 'ShotType'], axis=1)
    
    features = X.columns.tolist()
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return X, y, classes_legend, features


class MultinomialLogReg: 
    
    def __init__(self): 
        self.n_classes = None
        self.n_samples = None
        self.n_features = None
        
    def probabilities (self, X, betas):
        u = np.dot(X, betas.T)
        exp_u = np.exp(u)
        probs = exp_u / np.sum(exp_u, axis=1, keepdims=True)
        return probs
        
    def build(self, X, y):
        # one hot encoding of the target variable
        y = pd.get_dummies(y)
        y = np.array([y])[0]
        self.n_classes = y.shape[1]
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        betas_init = np.zeros((self.n_classes-1, self.n_features))
       
        def cost_func(betas, X, y):
            betas = np.hstack((betas, np.zeros(self.n_features)))
            betas = betas.reshape((self.n_classes, self.n_features))
            probs = self.probabilities(X, betas)
            masked_probs = probs[y == 1]
            log_likelihood = np.sum(np.log(masked_probs))
            #print(-log_likelihood)    
            return -log_likelihood

        betas = fmin_l_bfgs_b(func = cost_func,  
                              x0 = betas_init,
                              args=(X, y),
                              approx_grad = True)
        
        betas = np.array(betas[0])
        betas = np.hstack((betas, np.zeros(self.n_features)))
        betas = betas.reshape((self.n_classes, self.n_features))

        return MLRModel(betas, self.n_classes)
 
    
class MLRModel:
    
    def __init__(self, betas, n_classes):
        self.betas = betas
        self.n_classes = n_classes

    def predict(self, X):
        prob_matrix = MultinomialLogReg().probabilities(X, self.betas)
        
        return prob_matrix  



class OrdinalLogReg():
    
    def __init__(self): 
        self.n_classes = None
        self.n_samples = None
        self.n_features = None
        self.n_deltas = None
        self.thres = None
         
    def probabilities (self, y, thres, u):
        F1 = 1 / (1 + np.exp(-1 * (thres[int(y)+1] - u)))
        F2 = 1 / (1 + np.exp(-1 * (thres[int(y)] - u)))
        return F1 - F2
     
    def build(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_deltas = self.n_classes - 2
        params_init = np.zeros(self.n_deltas + self.n_features)

        def cost_func(params, X, y):
            deltas, betas = params[:self.n_deltas], params[self.n_deltas:]
            thres = np.zeros(self.n_classes + 1)
            thres[0], thres[1], thres[-1] = - np.inf, 0, np.inf
            for i in range(self.n_deltas):
                thres[i + 2] = thres[i + 1] + deltas[i]
            self.thres = thres
            u = np.dot(X, betas.T)
            probs = np.zeros(u.shape)
            for i in range(self.n_samples):
                probs[i] = self.probabilities(y[i], self.thres, u[i])
            log_likelihood = np.sum(np.log(probs))
            #print(-log_likelihood)    
            return -log_likelihood

        
        params = fmin_l_bfgs_b(func = cost_func,  
                              x0 = params_init,
                              args=(X, y),
                              bounds=[(1e-5, None) for i in range(self.n_deltas)] \
                                     + [(None, None) for i in range(self.n_features)],
                              approx_grad = True)
                              
        params = np.array(params[0])
        betas = params[self.n_deltas:]
   
        return OLRModel(betas, self.thres, self.n_classes)
    
    
class OLRModel:
    
    def __init__(self, betas, thres, n_classes):
        self.betas = betas
        self.thres = thres
        self.n_classes = n_classes

    def predict(self, X):
        prob_matrix = np.zeros((X.shape[0], self.n_classes))
        u = np.dot(X, self.betas.T)
        for i in range(X.shape[0]):
            for j in range(self.n_classes):
                prob = OrdinalLogReg().probabilities(j, self.thres, u[i])
                prob_matrix[i, j] = prob
        return prob_matrix


def compute_log_loss(y_test, probs):
    
    log_loss = 0
    for i in range(len(y_test)):
        prob = probs[i, int(y_test[i])]
        if prob == 0 or prob == 1 :
            eps = np.finfo(float).eps
            prob = np.clip(prob, eps, 1-eps)  
        log_loss += np.log(prob)
    log_loss = -log_loss / len(y_test)
    
    return log_loss


def bootstrap(X, y, seed):
    
    random.seed(seed)
    n_samples = X.shape[0]
    bootstrap_idx = random.choices(range(n_samples), k=n_samples)
    bootstrap_X = X[bootstrap_idx, :]
    bootstrap_y = y[bootstrap_idx]
    
    return bootstrap_X, bootstrap_y


def multinomial_bad_ordinal_good(n_samples, seed, n_classes=6, n_features=100):
    
    random.seed(seed)
    X = np.zeros((n_samples,n_features))  
    for i in range(n_features):
        X[:,i] = [random.gauss(0, 1) for j in range(n_samples)]
    
    y = np.random.choice(range(n_classes), size=n_samples)        
    
    return X, y


def multinomial_bad_ordinal_good2(n, rand):
    
    random.seed(rand)
    num_features = 90
    num_classes = 8
    X = np.random.normal(0, 1, (n, num_features))
    y = np.random.choice(range(num_classes), size=n)
        
    return X, y


if __name__ == '__main__':
       
    X, y, legend, features = load_data()
    X, y = np.array([X])[0], np.array([y])[0]  
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # start = time.time()
    # MLR_model = MultinomialLogReg().build(X_train, y_train)
    # end = time.time()
    # print('MLR Training Time:', end-start, 's')
    # MLR_probs = MLR_model.predict(X_test)
    # MLR_log_loss = compute_log_loss(y_test, MLR_probs)
    # print('MLR log loss:', MLR_log_loss)
    
    # # computing the accuracy
    # from sklearn.metrics import accuracy_score
    # MLR_preds = np.argmax(MLR_probs, axis=1)
    # max_probs = np.max(MLR_probs, axis=1)
    # MLR_accuracy = accuracy_score(y_test, MLR_preds)
    # print('MLR accuracy:', MLR_accuracy)
    
    # start = time.time()
    # OLR_model = OrdinalLogReg().build(X_train, y_train)
    # end = time.time()
    # print('OLR Training Time:', end-start, 's')
    # OLR_probs = OLR_model.predict(X_test)
    # OLR_log_loss = compute_log_loss(y_test, OLR_probs)
    # print('OLR log loss:', OLR_log_loss)
    
    # # computing the accuracy
    # OLR_preds = np.argmax(OLR_probs, axis=1)
    # max_probs = np.max(OLR_probs, axis=1)
    # OLR_accuracy = accuracy_score(y_test, OLR_preds)
    # print('OLR accuracy:', OLR_accuracy)

      
    # Multinomial Logistic Regression - Betas
    print('Computing coefficients of MLR model...')
    MLR_all_betas = []
    for i in range(100):
        print('Iteration:', i)
        start = time.time()
        X_train, y_train = bootstrap(X, y, i**2)
        MLR_model = MultinomialLogReg().build(X_train, y_train)
        end = time.time()
        print('MLR Training Time:', end-start, 's')
        MLR_all_betas.append(MLR_model.betas)
    
           
    MLR_betas = []
    for i in range(MLR_all_betas[0].shape[0]):
        betas_one_class = []
        for j in range(X.shape[1]):
            betas_one_feature = []
            for k in range(len(MLR_all_betas)):
                betas_one_feature.append(MLR_all_betas[k][i, j])
            betas_one_class.append(betas_one_feature)
        MLR_betas.append(betas_one_class)
    
    MLR_betas_mean = []
    MLR_betas_ci = []
    for i in range(len(MLR_betas)):
        MLR_betas_class_mean = []
        MLR_betas_class_ci = []
        for j in range(len(MLR_betas[i])):
           MLR_betas_class_mean.append(np.mean(MLR_betas[i][j]))
           MLR_betas_class_ci.append(1.96*np.std(MLR_betas[i][j])/np.sqrt(len(MLR_betas[i][j])))
        MLR_betas_mean.append(MLR_betas_class_mean)
        MLR_betas_ci.append(MLR_betas_class_ci)
    
    
    # Plotting MLR coefficients
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,20))
    class_names = list(legend.keys())[:5]
    
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[0,2]
    ax4 = axs[1,0]
    ax5 = axs[1,1]
    ax6 = axs[1,2]
    ax7 = axs[2,0]
    ax8 = axs[2,1]
    ax9 = axs[2,2]

    ax1.errorbar(np.arange(5), [MLR_betas_mean[i][0] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][0] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'blue')
    ax1.set_xticks(np.arange(5))
    ax1.set_title(features[0])
    ax1.set_ylabel('Coefficients')
    ax1.set_xticklabels(class_names, rotation=20)
    ax1.grid(True)
    
    ax2.errorbar(np.arange(5), [MLR_betas_mean[i][1] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][1] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'blue')
    ax2.set_xticks(np.arange(5))
    ax2.set_title(features[1])
    ax2.set_xticklabels(class_names, rotation=20)
    ax2.grid(True)
    
    ax3.errorbar(np.arange(5), [MLR_betas_mean[i][2] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][2] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'blue')
    ax3.set_xticks(np.arange(5))
    ax3.set_title(features[2])
    ax3.set_xticklabels(class_names, rotation=20)
    ax3.grid(True)
    
    ax4.errorbar(np.arange(5), [MLR_betas_mean[i][3] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][3] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'blue')
    ax4.set_xticks(np.arange(5))
    ax4.set_title(features[3])
    ax4.set_xticklabels(class_names, rotation=20)
    ax4.set_ylabel('Coefficients')
    ax4.grid(True)
    
    ax5.errorbar(np.arange(5), [MLR_betas_mean[i][4] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][4] for i in range(5)], color = 'blue',
                 fmt='o', markersize = 4, capsize=5, label = features[4])
    ax5.errorbar(np.arange(5), [MLR_betas_mean[i][5] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][5] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'red', label = features[5])
    ax5.set_xticks(np.arange(5))
    ax5.set_title('Competition')
    ax5.set_xticklabels(class_names, rotation=20)
    ax5.grid(True)
    ax5.legend()
    
    ax6.errorbar(np.arange(5), [MLR_betas_mean[i][6] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][6] for i in range(5)], color = 'blue',
                 fmt='o', markersize = 4, capsize=5, label = features[6])
    ax6.errorbar(np.arange(5), [MLR_betas_mean[i][7] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][7] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'red', label = features[7])
    ax6.set_xticks(np.arange(5))
    ax6.set_title('Competition')
    ax6.set_xticklabels(class_names, rotation=20)
    ax6.grid(True)
    ax6.legend()
    
    ax7.errorbar(np.arange(5), [MLR_betas_mean[i][8] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][8] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'blue', label = 'Player F')
    ax7.errorbar(np.arange(5), [MLR_betas_mean[i][9] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][9] for i in range(5)], 
                 fmt='o', markersize = 4, capsize=5, color = 'red', label = 'Player G')
    ax7.set_xticks(np.arange(5))
    ax7.set_title('Player Type')
    ax7.set_xticklabels(class_names, rotation=20)
    ax7.grid(True)
    ax7.set_ylabel('Coefficients')
    ax7.legend()
    
    ax8.errorbar(np.arange(5), [MLR_betas_mean[i][10] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][10] for i in range(5)], color = 'blue', 
                 fmt='o', markersize = 4, capsize=5, label = features[10])
    ax8.set_xticks(np.arange(5))
    ax8.set_title('Movement: drive')
    ax8.set_xticklabels(class_names, rotation=20)
    ax8.grid(True)
    ax8.legend()
    
    ax9.errorbar(np.arange(5), [MLR_betas_mean[i][11] for i in range(5)], 
                 yerr=[MLR_betas_ci[i][11] for i in range(5)], color = 'blue', 
                 fmt='o', markersize = 4, capsize=5)
    ax9.set_xticks(np.arange(5))
    ax9.set_title('Movement: no')
    ax9.set_xticklabels(class_names, rotation=20)
    ax9.grid(True)
    
    plt.show()       
    fig.savefig('multinomial.png', dpi=300)
    


    # Ordinal Logistic Regression - Betas
    print('Computing coefficients of OLR model...')
    OLR_all_betas = []
    OLR_all_thres = []
    for i in range(100):
        print('Iteration:', i)
        X_train, y_train = bootstrap(X, y, i**2)
        OLR_model = OrdinalLogReg().build(X_train, y_train)
        OLR_all_betas.append(OLR_model.betas)
        OLR_all_thres.append(OLR_model.thres)
 
        
    OLR_betas_mean = []
    OLR_betas_ci = []
    OLR_betas = []
    for i in range(len(OLR_all_betas[0])):
        betas_one_feature = []
        for j in range(len(OLR_all_betas)):
            betas_one_feature.append(OLR_all_betas[j][i])
        OLR_betas.append(betas_one_feature)
        OLR_betas_mean.append(np.mean(betas_one_feature))   
        OLR_betas_ci.append(1.96*np.std(betas_one_feature)/np.sqrt(len(betas_one_feature)))
    
    OLR_thres_mean = []
    OLR_thres_ci = []
    for i in range(1, len(OLR_all_thres[0])-1):
        one_thres = []
        for j in range(len(OLR_all_betas)):
            one_thres.append(OLR_all_thres[j][i])
        OLR_thres_mean.append(np.mean(one_thres))
        OLR_thres_ci.append(1.96*np.std(one_thres)/np.sqrt(len(one_thres)))
        
            
    # Plotting OLR coefficients
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(OLR_betas_mean, np.arange(len(OLR_betas_mean)),
                xerr=OLR_betas_ci, fmt='o', markersize = 4, capsize=6, color='blue')
    ax.set_yticks(np.arange(len(OLR_betas_mean)))
    ax.set_yticklabels(features)
    ax.grid(True)
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Feature')
    plt.show()
    fig.savefig('ordinal.png', dpi=300) 
         


    # Multiomial bad ordinal good dataset
    
    print('Testing models on MBOG dataset...')
    MBOG_MLR_log_loss = [] 
    MBOG_OLR_log_loss = []
       
    for i in range(2,102): 
        
        print('Iteration:', i-1)
        MBOG_X_train, MBOG_y_train = multinomial_bad_ordinal_good(MBOG_TRAIN, i**2)
        MBOG_X_test, MBOG_y_test = multinomial_bad_ordinal_good(1000, i**3)
        MBOG_MLR_model = MultinomialLogReg().build(MBOG_X_train, MBOG_y_train)
        MBOG_MLR_probs = MBOG_MLR_model.predict(MBOG_X_test)
        MBOG_log_loss = compute_log_loss(MBOG_y_test, MBOG_MLR_probs)
        MBOG_MLR_log_loss.append(MBOG_log_loss)
        MBOG_OLR_model = OrdinalLogReg().build(MBOG_X_train, MBOG_y_train)
        MBOG_OLR_probs = MBOG_OLR_model.predict(MBOG_X_test)
        MBOG_log_loss = compute_log_loss(MBOG_y_test, MBOG_OLR_probs)
        MBOG_OLR_log_loss.append(MBOG_log_loss)
        
    print('Mean MLR Log-Loss:', np.mean(MBOG_MLR_log_loss), 'SE:', np.std(MBOG_MLR_log_loss, ddof=1)/np.sqrt(len(MBOG_MLR_log_loss)))
    print('Mean OLR Log-Loss:', np.mean(MBOG_OLR_log_loss), 'SE:', np.std(MBOG_OLR_log_loss, ddof=1)/np.sqrt(len(MBOG_MLR_log_loss)))    
    

