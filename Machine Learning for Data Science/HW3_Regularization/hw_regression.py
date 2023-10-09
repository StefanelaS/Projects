import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
plt.style.use('ggplot')


class RidgeReg:
    def __init__(self, lmbd):
        self.lmbd = lmbd
        self.coeff = None

    def fit(self, X, y):
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        I = np.eye(X.shape[1])
        I[0] = 0 
        self.coeff = np.linalg.solve(np.dot(X.T, X) + self.lmbd*I, np.dot(X.T, y))

    def predict(self, X):
        X = np.array(X)
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        return np.dot(X, self.coeff)


class LassoReg:
    def __init__(self, lmbd):
       self.lmbd = lmbd
       self.coeff = None
       self.intercept = None

    def fit(self, X, y):
        
        def function(w):
            self.coeff = w[:-1]
            self.intercept = w[-1]
            return np.sum((np.dot(X, self.coeff) + self.intercept - y) ** 2)  + self.lmbd * np.sum(np.abs(self.coeff))
        
        w0 = np.zeros(X.shape[1] + 1)
        res = minimize(function, w0, method='Powell')
        self.coeff = res.x[:-1]
        self.intercept_ = res.x[-1]
     
    def predict(self, X):
        X = np.array(X)
        intercept_col = np.ones(X.shape[0])
        X = np.insert(X, 0, intercept_col, axis=1)
        self.coeff = np.insert(self.coeff, 0, self.intercept)
        return np.dot(X, self.coeff)


class RegularizationTest(unittest.TestCase):

    def test_ridge_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], 30, delta=0.1)
        self.assertAlmostEqual(y[1], 50, delta=0.1)
        
    def test_ridge_two_features(self):
        X = np.array([[1, 5],
                      [10, 10],
                      [100, 15],
                      [1000, 20]])
        y = 1 + X[:,0] + 2*X[:,1] 
        model = LassoReg(1)
        model.fit(X, y)
        y = model.predict([[10, 10],
                           [20, 5]])
        self.assertAlmostEqual(y[0], 31, delta=0.1)
        self.assertAlmostEqual(y[1], 31, delta=0.1)
     
    def test_lasso_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = LassoReg(1)
        model.fit(X, y)
        y = model.predict([[10],
                           [20]])
        self.assertAlmostEqual(y[0], 30, delta=0.1)
        self.assertAlmostEqual(y[1], 50, delta=0.1)
    
    def test_lasso_two_features(self):
        X = np.array([[1, 5],
                      [10, 10],
                      [100, 15],
                      [1000, 20]])
        y = 1 + X[:,0] + 2*X[:,1] 
        model = LassoReg(1)
        model.fit(X, y)
        y = model.predict([[10, 10],
                           [20, 5]])
        self.assertAlmostEqual(y[0], 31, delta=0.1)
        self.assertAlmostEqual(y[1], 31, delta=0.1)


def load(fname):
    data = np.genfromtxt('superconductor.csv', dtype=float, delimiter=',', skip_header=True) 
    X_train = data[:200, :-1]
    y_train = data[:200, -1]
    X_test = data[200:, :-1]
    y_test = data[200:, -1]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return data, X_train, y_train, X_test, y_test


def cross_validation(model, X, y, k, seed = 42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rmse_scores = []
    complexities = []
    for train_index, val_index in kf.split(X):
       X_train, y_train = X[train_index], y[train_index]
       X_val, y_val = X[val_index], y[val_index]
       model.fit(X_train, y_train)
       complexity = sum([1 for coeff in model.coeff if coeff != 0])
       complexities.append(complexity)
       y_pred = model.predict(X_val)
       rmse = np.sqrt(np.mean((y_pred - y_val)**2))
       rmse_scores.append(rmse)
    return np.mean(rmse_scores), np.mean(complexities)
    

def superconductor(X_train, y_train, X_test, y_test):

    lambdas = np.geomspace(0.001, 100, num=1000)
    rmse_scores = np.zeros(len(lambdas))
    complexities = np.zeros(len(lambdas))
    for i in range(0,len(lambdas)):
        model = RidgeReg(lambdas[i])
        rmse, complexity = cross_validation(model, X_train, y_train, 10)
        rmse_scores[i] = rmse
    
    plt.figure()
    plt.semilogx(lambdas, rmse_scores)
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
    
    lowest_rmse = min(rmse_scores)
    best_idx = np.argmin(rmse_scores)
    best_lambda_ridge = lambdas[best_idx]
    
    print('RIDGE REGRESSION')
    print('Optimal regularization weight:') 
    print( 'Lambda', best_lambda_ridge, 'RMSE', lowest_rmse)


    lambdas = np.arange(0, 1001, 10)
    rmse_scores = np.zeros(len(lambdas))
    complexities = np.zeros(len(lambdas))
    for i in range(0,len(lambdas)):
        model = LassoReg(lambdas[i])
        rmse, complexity = cross_validation(model, X_train, y_train, 10)
        rmse_scores[i] = rmse
        complexities[i] = int(complexity)
        print(i)
    
    plt.figure()
    plt.plot(lambdas, complexities)
    plt.xlabel('Lambda')
    plt.ylabel('Complexity')
    
    plt.figure()
    plt.plot(complexities, rmse_scores)
    plt.xlabel('Complexity')
    plt.ylabel('RMSE')
    
    plt.figure()
    plt.plot(lambdas, rmse_scores)
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
   
    best_idx = np.argmin(rmse_scores)
    lowest_rmse = min(rmse_scores)
    best_lambda = lambdas[best_idx]
    best_lambda_compl = complexities[best_idx]    
    rmse_std = np.std(rmse_scores)
    upper_limit = lowest_rmse + rmse_std
    lower_limit = lowest_rmse - rmse_std
    within_range = np.logical_and(rmse_scores >= lower_limit, rmse_scores <= upper_limit)
    within_range_scores = rmse_scores[within_range]
    within_range_lambdas = lambdas[within_range]
    within_range_complexities = complexities[within_range]
    chosen_lambda_idx = np.argmax(within_range_lambdas)
    chosen_lambda = within_range_lambdas[chosen_lambda_idx]
    chosen_lambda_rmse = within_range_scores[chosen_lambda_idx]
    
    print('LASSO REGRESSION')
    print('Regularization weight with lowest RMSE:') 
    print( 'Lambda', best_lambda, 'RMSE', lowest_rmse, 'Complexity', best_lambda_compl)
    print('Optimal regularization weight:')
    print('Lambda', chosen_lambda, 'RMSE', chosen_lambda_rmse, 'Complexity', min(within_range_complexities))
   
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    print('Linear Regression RMSE:', rmse)
    
    ridge_model = RidgeReg(lmbd=best_lambda_ridge)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    rmse_ridge = np.sqrt(np.mean((y_pred - y_test)**2))
    print('Ridge Regression RMSE:', rmse_ridge)
    
    lasso_model = LassoReg(lmbd=chosen_lambda)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    rmse_lasso = np.sqrt(np.mean((y_pred - y_test)**2))
    print('Lasso Regression RMSE:', rmse_lasso)
    
    
if __name__ == "__main__":
    data, X_train, y_train, X_test, y_test = load("superconductor.csv")
    superconductor(X_train, y_train, X_test, y_test)
    unittest.main()
