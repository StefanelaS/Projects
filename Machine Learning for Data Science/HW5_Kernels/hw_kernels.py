# -*- coding: utf-8 -*-
"""
Created on Sat May 20 06:43:16 2023

@author: Asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# Function that computes MSE
def MSE(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Function that finds best parameters based on MSE and returns best parameters, predictions and MSE
def get_best_fit(X, y, model, kernel, kernel_params, lambdas):
    best_mse = 10000
    best_kernel_param = 0
    best_lambda = 0
    preds = None
    support_vectors = None
    for i in range(len(kernel_params)):
        for j in range(len(lambdas)):
            if model == 'KRR' and kernel == 'polynomial':
                fitter = KernelizedRidgeRegression(kernel=Polynomial(M=kernel_params[i]), lambda_=lambdas[j])
            elif model == 'KRR' and kernel == 'RBF':
                fitter = KernelizedRidgeRegression(kernel=RBF(sigma=kernel_params[i]), lambda_=lambdas[j])
            elif model == 'SVR' and kernel == 'polynomial':
                fitter = SVR(kernel=Polynomial(M=kernel_params[i]), lambda_=lambdas[j])
            elif model == 'SVR' and kernel == 'RBF':
                fitter = SVR(kernel=RBF(sigma=kernel_params[i]), lambda_=lambdas[j])
            m = fitter.fit(X, y)
            y_pred = m.predict(X)
            mse = MSE(y, y_pred)
            if mse < best_mse:
                best_kernel_param = kernel_params[i]
                best_lambda = lambdas[j]
                preds = y_pred
                best_mse = mse
                if model == 'SVR':
                    support_vectors = m.get_support_vectors()
    return best_kernel_param, best_lambda, preds, best_mse, support_vectors


def get_best_lambdas(X_train, y_train, X_test, y_test, model, kernel, kernel_params, lambdas):
    mse_scores = []
    best_lambdas = []
    best_lambda = 0
    n_support_vectors = []
    support_vectors = None
    for i in range(len(kernel_params)):
        best_mse = 100000
        for j in range(len(lambdas)):
            if model == 'KRR' and kernel == 'polynomial':
                fitter = KernelizedRidgeRegression(kernel=Polynomial(M=kernel_params[i]), lambda_=lambdas[j])
            elif model == 'KRR' and kernel == 'RBF':
                fitter = KernelizedRidgeRegression(kernel=RBF(sigma=kernel_params[i]), lambda_=lambdas[j])
            elif model == 'SVR' and kernel == 'polynomial':
                fitter = SVR(kernel=Polynomial(M=kernel_params[i]), lambda_=lambdas[j], epsilon = 8)
            else:
                fitter = SVR(kernel=RBF(sigma=kernel_params[i]), lambda_=lambdas[j], epsilon = 8)
            m = fitter.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            mse = MSE(y_test, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_lambda = lambdas[j]
                if model == 'SVR':
                    support_vectors = len(m.get_support_vectors())
        mse_scores.append(best_mse)
        best_lambdas.append(best_lambda)
        n_support_vectors.append(support_vectors)
        
    return mse_scores, best_lambdas, n_support_vectors 


# Polynomial Kernel
class Polynomial:
    
    def __init__(self, M):
        self.M = M
    
    def __call__(self, A, B):
        
        if A.ndim == 1 and B.ndim == 1:
            return (np.dot(A, B.T) + 1) ** self.M
        
        elif A.ndim == 2 and B.ndim == 2:
            if A.shape[1] == B.shape[0]:
                return (np.dot(A, B) + 1) ** self.M
            elif A.shape[1] == B.shape[1]:
                return (np.dot(A, B.T) + 1) ** self.M
            else:
                raise ValueError("Invalid input dimensions for Polynomial kernel.")
         
        elif A.ndim == 1 and B.ndim == 2:
          if A.shape[0] == B.shape[0]:
              return (np.dot(A, B) + 1) ** self.M
          elif A.shape[0] == B.shape[1]:
              return (np.dot(A, B.T) + 1) ** self.M
          else:
             raise ValueError("Invalid input dimensions for Polynomial kernel.")      
                         
        elif A.ndim == 2 and B.ndim == 1:
            if A.shape[1] == B.shape[0]:
                return (np.dot(A, B) + 1) ** self.M
            elif A.shape[0] == B.shape[0]:
                return (np.dot(A.T, B) + 1) ** self.M
            else:
                raise ValueError("Invalid input dimensions for Polynomial kernel.")
        
        else:
            raise ValueError("Invalid input dimensions for Polynomial kernel.")


# RBF kernel 
class RBF:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def __call__(self, A, B):
        D = 2*np.square(self.sigma)
        if A.ndim == 1 and B.ndim == 1:
            return np.exp(-np.sum(np.square(A-B))/D)
        
        elif A.ndim == 1 and B.ndim == 2:
            if A.shape[0] == B.shape[0]:
                return np.exp(-np.sum(np.square(A-B.T), axis=1)/D)
            elif A.shape[0] == B.shape[1]:
                return np.exp(-np.sum(np.square(A-B), axis=1)/D)
            else:
               raise ValueError("Invalid input dimensions for RBF kernel.") 
               
        elif A.ndim == 2 and B.ndim == 1:
            if A.shape[0] == B.shape[0]:
                return np.exp(-np.sum(np.square(A.T-B), axis=1)/D)
            elif A.shape[1] == B.shape[0]:
                return np.exp(-np.sum(np.square(A-B), axis=1)/D)
            else:
               raise ValueError("Invalid input dimensions for RBF kernel.")  
               
        elif A.ndim == 2 and B.ndim == 2:
            if A.shape[1] == B.shape[0]:
                dist = np.sum(np.square(A), axis=1, keepdims=True) - 2 * np.dot(A, B) + np.sum(np.square(B.T), axis=1)
                return np.exp(-dist/D)
            elif A.shape[1] == B.shape[1]:
                dist = np.sum(np.square(A), axis=1, keepdims=True) - 2 * np.dot(A, B.T) + np.sum(np.square(B), axis=1)
                return np.exp(-dist/D)
            else:
                raise ValueError("Invalid input dimensions for RBF kernel.")
                
        else:
            raise ValueError("Invalid input dimensions for RBF kernel.")
            

class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.X = X
        K = self.kernel.__call__(self.X, self.X)
        inverse = np.linalg.inv(K + self.lambda_ * np.eye(self.X.shape[0]))
        coeffs = np.dot(inverse, y)
        return KRRModel(coeffs, self.kernel, self.X)
    
class KRRModel:
    def __init__(self, coeffs, kernel, X_train):
        self.coeffs = coeffs
        self.kernel = kernel
        self.X_train = X_train
    
    def predict(self, X):
        K = self.kernel.__call__(X, self.X_train)
        return np.dot(K, self.coeffs)



class SVR:
    def __init__(self, kernel, lambda_, epsilon=0.5):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X = X
        self.y = y
        K = self.kernel.__call__(X, X)
        C = 1 / self.lambda_
 
        P = np.kron(K, np.array([[1, -1], [-1, 1]]))
        n_samples = P.shape[0]
        P = cvxopt.matrix(P, (n_samples, n_samples), 'd')

        q = np.zeros(n_samples)
        q[::2] = self.epsilon - self.y
        q[1::2] = self.epsilon + self.y
        q = cvxopt.matrix(q, (n_samples, 1), 'd')

        G = np.vstack([np.eye(n_samples), -np.eye(n_samples)])
        G = cvxopt.matrix(G, (2 * n_samples, n_samples), 'd')
        
        h = np.hstack((C * np.ones(n_samples), np.zeros(n_samples)))
        h = cvxopt.matrix(h, (2 * n_samples, 1), 'd')

        A = np.ones(n_samples)
        A = np.where(np.arange(n_samples) % 2 == 1, -1.0, A)
        A = cvxopt.matrix(A, (1, n_samples), 'd')
        b = cvxopt.matrix(0, (1, 1), 'd')

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        return SVRModel(self.X, self.y, self.kernel, sol)


class SVRModel:
    def __init__(self, X_train, y_train, kernel, sol):
        self.X_train = X_train
        self.y_train =  y_train
        self.kernel = kernel
        self.sol = sol
        
    def get_alpha(self):
        sol_x = np.array(self.sol['x'])
        self.alphas = sol_x[::2, 0]
        self.alphas_star = sol_x[1::2, 0]
        return np.vstack([self.alphas, self.alphas_star]).T 
    
    def get_b(self):
        return np.array(self.sol['y'])[0]
    
    def get_support_vectors(self, tolerance=1e-05):
        self.get_alpha()
        vector_idx = np.where((np.abs(self.alphas) > tolerance) | (np.abs(self.alphas_star) > tolerance))[0]
        support_vectors_x = self.X_train[vector_idx]
        support_vectors_y = self.y_train[vector_idx].reshape(-1, 1)
        return np.hstack([support_vectors_x, support_vectors_y])
    
    def predict(self, X):
        self.get_alpha()
        K = self.kernel.__call__(X, self.X_train)
        b = self.get_b()
        y_pred = ((self.alphas - self.alphas_star) * K).sum(axis=1) + b
        return y_pred
 
    
 
if __name__ == "__main__":
    
    # SINE DATA
    sine_df = pd.read_csv('sine.csv')
    X = sine_df['x'].values
    y = sine_df['y'].values
    
    # standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    sine_df['x_norm'] = X

    # define candidate parameters
    M = [i for i in range(1,16)]
    lambdas = [i/10 for i in range(2,10)]
    sigmas = [i/10 for i in range(1,20,2)]
    epsilon = 0.5
    
    # KRR + polynomial kernel
    kernel_param, lmbd, preds, mse, _ = get_best_fit(X, y, 'KRR', 'polynomial', M, lambdas) 
    print('KRR + polynomial kernel:', 'M = {}, '.format(kernel_param), 
          'lambda = {}, '.format(lmbd), 'MSE = {}'.format(mse))
    sine_df['KRR_poly'] = preds

    
    # KRR + RBF kernel
    kernel_param, lmbd, preds, mse, _ = get_best_fit(X, y, 'KRR','RBF', sigmas, lambdas) 
    print('KRR + RBF kernel:', 'sigma = {}, '.format(kernel_param), 
          'lambda = {}, '.format(lmbd), 'MSE = {}'.format(mse))
    sine_df['KRR_RBF'] = preds
    
   
    # SVR + poly kernel
    kernel_param, lmbd, preds, mse, sv_poly = get_best_fit(X, y, 'SVR', 'polynomial', M, lambdas) 
    print('SVR + polynomial kernel:', 'M = {}, '.format(kernel_param), 
          'lambda = {}, '.format(lmbd), 'MSE = {}'.format(mse))
    sine_df['SVR_poly'] = preds
    
  
    # SVR + RBF kernel
    kernel_param, lmbd, preds, mse, sv_rbf = get_best_fit(X, y, 'SVR', 'RBF', sigmas, lambdas) 
    print('SVR + RBF kernel:', 'sigma = {}, '.format(kernel_param), 
          'lambda = {}, '.format(lmbd), 'MSE = {}'.format(mse))
    sine_df['SVR_RBF'] = preds

 
    # Plotting
    sorted_df = sine_df.sort_values('x_norm').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(7,4))
    plt.scatter(sorted_df['x_norm'], sorted_df['y'], marker = 'o', s = 20, 
                label = 'standardized sine data', color = 'gray')
    plt.plot(sorted_df['x_norm'], sorted_df['KRR_poly'], color = 'red', 
             linewidth=2, label = 'polynomial kernel fit')
    plt.plot(sorted_df['x_norm'], sorted_df['KRR_RBF'], color = 'black', 
             linewidth=2, label = 'RBF kernel fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=7, loc = 'upper left')
    plt.savefig('KRR_sine.png', dpi=300)
    plt.show()
 

    fig, ax = plt.subplots(figsize=(7,4))
    plt.scatter(sorted_df['x_norm'], sorted_df['y'], marker = 'o', s = 17, 
                label = 'standardized sine data', color = 'gray')
    plt.scatter(sv_poly[:, 0], sv_poly[:, 1], marker = 's', s = 20, 
                color = 'darkorange', label = 'support vectors')
    plt.plot(sorted_df['x_norm'], sorted_df['SVR_poly'], color = 'black', 
             linewidth=2, label = 'polynomial kernel fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=7, loc = 'upper left')
    plt.savefig('SVR_sine_poly.png', dpi=300)
    plt.show()
  

    fig, ax = plt.subplots(figsize=(7,4))
    plt.scatter(sorted_df['x_norm'], sorted_df['y'], marker = 'o', s = 17, 
                label = 'standardized sine data', color = 'gray')
    plt.scatter(sv_rbf[:, 0], sv_rbf[:, 1], marker = 's', s = 20,
                color = 'darkorange', label = 'support vectors')
    plt.plot(sorted_df['x_norm'], sorted_df['SVR_RBF'], color = 'black', 
             linewidth=2, label = 'RBF kernel fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=7, loc = 'upper left')
    plt.savefig('SVR_sine_RBF.png', dpi=300)
    plt.show()
 
    
    # HOUSING DATA
    housing_df = pd.read_csv('housing2r.csv')
    
    # separate training and validation sets
    X_housing = housing_df.iloc[:, : -1].values
    y_housing = housing_df['y'].values
    train_size = int(0.8 * len(housing_df))
    X_train, y_train = X_housing[: train_size], y_housing[: train_size]
    X_test, y_test = X_housing[train_size :], y_housing[train_size :]
    
    # standardizing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    M = [i for i in range(1,11)]
    lambdas = [0.1, 0.5, 1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 500, 1000]
    sigmas = [i for i in range(1,11)]

    # KRR + polynomial
    krr_poly_mse = []
    for m in M:
        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=m), lambda_=1)
        m = fitter.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mse = MSE(y_test, y_pred)
        krr_poly_mse.append(mse)
        
    krr_poly_mse_opt, krr_poly_best_lambda, _ = get_best_lambdas(X_train, y_train, 
                                                          X_test, y_test, 'KRR', 
                                                         'polynomial', M, lambdas)
    # KRR + RBF
    krr_RBF_mse = []
    for sigma in sigmas:
        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=1)
        m = fitter.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mse = MSE(y_test, y_pred)
        krr_RBF_mse.append(mse)
    
    krr_RBF_mse_opt, krr_RBF_best_lambda, _ = get_best_lambdas(X_train, y_train, 
                                                          X_test, y_test, 'KRR', 
                                                         'RBF', sigmas, lambdas)
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    
    axs[0].plot(M[:-1], krr_poly_mse[:-1], label = 'MSE with λ = 1', color = 'red')
    axs[0].plot(M[:-1], krr_poly_mse_opt[:-1], label = 'MSE with optimal λ', color = 'black')
    axs[0].set_xlabel('polynomial degree M')
    axs[0].set_ylabel('MSE')
    axs[0].set_xticks(M[:-1])
    axs[0].legend()
    axs[0].set_title('a) KRR + Polynomial Kernel')
    
    axs[1].plot(sigmas, krr_RBF_mse, label = 'MSE with λ = 1', color = 'red')
    axs[1].plot(sigmas, krr_RBF_mse_opt, label = 'MSE with optimal λ', color = 'black')
    axs[1].set_xlabel('parameter σ')
    axs[1].set_ylabel('MSE')
    axs[1].set_xticks(sigmas)
    axs[1].legend()
    axs[1].set_title('b) KRR + RBF Kernel')
    plt.savefig('KRR_housing.png', dpi=300)
    plt.show()
    
    
    # SVR + Polynomial
    svr_poly_mse = []
    svr_poly_sv = []
    for m in M:
        fitter = SVR(kernel=Polynomial(M=m), lambda_=1, epsilon = 8)
        m = fitter.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        n_sv = len(m.get_support_vectors())
        svr_poly_sv.append(n_sv)
        mse = MSE(y_test, y_pred)
        svr_poly_mse.append(mse)
  
    svr_poly_mse_opt, svr_poly_best_lambda, n_sv_poly = get_best_lambdas(X_train, y_train, 
                                                          X_test, y_test, 'SVR', 
                                                         'polynomial', M, lambdas) 
   
    # SVR + RBF
    svr_RBF_mse = []
    svr_RBF_sv = []
    for sigma in sigmas:
        fitter = SVR(kernel=RBF(sigma=sigma), lambda_=1, epsilon = 8)
        m = fitter.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        n_sv = len(m.get_support_vectors())
        svr_RBF_sv.append(n_sv)
        mse = MSE(y_test, y_pred)
        svr_RBF_mse.append(mse)
       
    svr_RBF_mse_opt, svr_RBF_best_lambda, n_sv_RBF = get_best_lambdas(X_train, y_train, 
                                                          X_test, y_test, 'SVR', 
                                                         'RBF', sigmas, lambdas)
    
    # Plotting
    fig, axs = plt.subplots(2,1, figsize=(6,12))
    
    axs[0].plot(M[:-1], svr_poly_mse[:-1], label = 'MSE (λ = 1)', color = 'red')
    axs[0].scatter(M[:-1], svr_poly_mse[:-1], label = 'No. support vectors (λ = 1)', 
                   color = 'red', marker = 'o')
    axs[0].plot(M[:-1], svr_poly_mse_opt[:-1], label = 'MSE (optimal λ)', color = 'black')
    axs[0].scatter(M[:-1], svr_poly_mse_opt[:-1], label = 'No. support vectors (optimal λ)',
                   color = 'black', marker = 'o')
    axs[0].set_xlabel('polynomial degree M')
    axs[0].set_ylabel('MSE')
    axs[0].set_xticks(M[:-1])
    axs[0].legend(fontsize=7.5, loc = 'upper left')
    axs[0].set_title('a) SVR + Polynomial Kernel')
    for i, n_sv in enumerate(svr_poly_sv):
        axs[0].text(M[i], svr_poly_mse[i], '{}'.format(n_sv), 
                   fontsize = 10, color = 'red', va = 'bottom', ha = 'right')
    for i, n_sv in enumerate(n_sv_poly):
        axs[0].text(M[i], svr_poly_mse_opt[i], '{}'.format(n_sv), 
                    fontsize = 10, color = 'black', va = 'top', ha = 'left')
    
    axs[1].plot(sigmas, svr_RBF_mse, label = 'MSE (λ = 1)', color = 'red')
    axs[1].scatter(sigmas, svr_RBF_mse, label = 'No. support vectors (λ = 1)', 
                   color = 'red', marker = 'o')
    axs[1].plot(sigmas, svr_RBF_mse_opt, label = 'MSE (optimal λ)', color = 'black')
    axs[1].scatter(M, svr_RBF_mse_opt, label = 'No. support vectors (optimal λ)', 
                   color = 'black', marker = 'o')
    axs[1].set_xlabel('parameter σ')
    axs[1].set_ylabel('MSE')
    axs[1].set_xticks(sigmas)
    axs[1].legend(fontsize=7.5, loc = 'upper left')
    axs[1].set_title('b) SVR + RBF Kernel')
    for i, n_sv in enumerate(svr_RBF_sv):
        axs[1].text(sigmas[i], svr_RBF_mse[i], '{}'.format(n_sv), 
                   fontsize = 10, color = 'red', va = 'bottom', ha = 'right')
    for i, n_sv in enumerate(n_sv_RBF):
        axs[1].text(sigmas[i], svr_RBF_mse_opt[i], '{}'.format(n_sv), 
                   fontsize = 10, color = 'black', va = 'top', ha = 'left')
    plt.savefig('SVR_housing_3.png', dpi=300)
    plt.show()