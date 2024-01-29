# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 04:28:03 2023

@author: Asus
"""

import numpy as np
import math
import sympy as sp
import random
import matplotlib.pyplot as plt
from ex4_utils import kalman_step


def get_symbolic_matrices(model):
    
    if model == 'RW':
        F = np.zeros((2,2))
        L = np.identity(2)
        H = np.identity(2)
        
    elif model == 'NCV':
        F = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        L = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        H = np.array([[1, 0 ,0, 0], [0, 1, 0, 0]])
        
    elif model == 'NCA':
        F = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        L = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        
    T, qi = sp.symbols('T'), sp.symbols('qi')
    F, L = sp.Matrix(F), sp.Matrix(L)
    
    A = (sp.exp(F * T)).subs(T, 1)
    A = np.array(A, dtype="float")
    
    Q = (sp.integrate((A * L) * qi * (A * L).T, (T, 0, T))).subs(T, 1)
    
    return A, Q, H


def get_matrices_values(r, q, Q):
    
    qi = sp.symbols('qi')
    Q = Q.subs(qi, q)
    Q = np.array(Q, dtype="float")
    
    R = r * np.identity(2)
    R = np.array(R, dtype="float")
    
    return Q, R


def kalman_filter(x, y, A, Q, H, R):
    
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]

    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye(A.shape[0], dtype=np.float32)

    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, H, Q, R, np.reshape(np.array([x[j] , y [j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
        sx[j] = state[0]
        sy[j] = state[1]
    
    return sx, sy


# Plotting the curves

# curve 1 - spiral curve
N = 40
v = np.linspace(5 * math.pi, 0, N)
x = np.cos(v) * v
y = np.sin(v) * v

params = [1, 1, 1, 5, 100]
models = ['RW', 'NCV', 'NCA']

fig, axs = plt.subplots(3, 5, figsize=(15, 12))

for i in range(len(models)):
    for j in range(len(params)):
        
        A, Q, H = get_symbolic_matrices(models[i])
        Q, R = get_matrices_values(params[j], params[-(j+1)], Q)
        sx, sy = kalman_filter(x, y, A, Q, H, R)
        
        axs[i, j].plot(x, y, c="red", linewidth=1)
        axs[i,j].scatter(x, y, facecolors='none', edgecolors='red')
        axs[i,j].plot(sx, sy, c="blue", linewidth=1)
        axs[i,j].scatter(sx, sy, facecolors='none', edgecolors='blue')
        axs[i,j].title.set_text(models[i] + ": r = {}, q = {}".format(params[j], params[-(j+1)]))

plt.show()   
fig.savefig('curve1.png', dpi=300)   


# curve 2 - jagged curve

R, r, k = 2, 1, 2

v = np.linspace(0, 2 * math.pi, N)
x = (R - r) * np.cos(v) + r * np.cos(k * v) + np.random.normal(scale=0.11, size=N)
y = (R - r) * np.sin(v) - r * np.sin(k * v) + np.random.normal(scale=0.11, size=N)

fig, axs = plt.subplots(3, 5, figsize=(15, 12))

for i in range(len(models)):
    for j in range(len(params)):
        
        A, Q, H = get_symbolic_matrices(models[i])
        Q, R = get_matrices_values(params[j], params[-(j+1)], Q)
        sx, sy = kalman_filter(x, y, A, Q, H, R)
        
        axs[i, j].plot(x, y, c="red", linewidth=1)
        axs[i,j].scatter(x, y, facecolors='none', edgecolors='red')
        axs[i,j].plot(sx, sy, c="blue", linewidth=1)
        axs[i,j].scatter(sx, sy, facecolors='none', edgecolors='blue')
        axs[i,j].title.set_text(models[i] + ": r = {}, q = {}".format(params[j], params[-(j+1)]))
        
plt.show()
fig.savefig('curve2.png', dpi=300) 


# curve 3 - complex star

R, r, k = 9, 8, 10 

v = np.linspace(0, 2 * math.pi, N)
x = (R - r) * np.cos(v) + r * np.cos(k * v)
y = (R - r) * np.sin(v) - r * np.sin(k * v)

fig, axs = plt.subplots(3, 5, figsize=(15, 12))

for i in range(len(models)):
    for j in range(len(params)):
        
        A, Q, H = get_symbolic_matrices(models[i])
        Q, R = get_matrices_values(params[j], params[-(j+1)], Q)
        sx, sy = kalman_filter(x, y, A, Q, H, R)
        
        axs[i, j].plot(x, y, c="red", linewidth=1)
        axs[i,j].scatter(x, y, facecolors='none', edgecolors='red')
        axs[i,j].plot(sx, sy, c="blue", linewidth=1)
        axs[i,j].scatter(sx, sy, facecolors='none', edgecolors='blue')
        axs[i,j].title.set_text(models[i] + ": r = {}, q = {}".format(params[j], params[-(j+1)]))
        
plt.show()
fig.savefig('curve3.png', dpi=300) 

