import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

def rbf_1d(xx, cc, hh = 1):
    return np.exp(-np.power((xx-cc), 2) / hh**2)

def phi_rbf(xx):
    temp = np.array([rbf_1d(xx, 1), rbf_1d(xx, 2), rbf_1d(xx, 3), rbf_1d(xx, 4), rbf_1d(xx, 5), rbf_1d(xx, 6)])[:,:,0].T
    return np.concatenate([np.ones((temp.shape[0],1)), temp], axis=1)

def phi_quad(xx):
    temp = np.array([np.power(xx, 1), np.power(xx, 2), np.power(xx, 3), np.power(xx, 4)])[:,:,0].T
    return np.concatenate([np.ones((temp.shape[0],1)), temp], axis=1)

def phi_lin(xx):
    return np.concatenate([np.ones((xx.shape[0],1)), xx], axis=1)
    
N = 10
X = np.linspace(0, 10, num=N).reshape(N, 1)
yy = rbf_1d(X.reshape(N,), 5) + 0.05*np.random.randn(N)

x_grid = np.linspace(0, 10, num=N*N).reshape(N*N, 1)
yy_true = rbf_1d(x_grid.reshape(N*N,), 5)

plt.clf()
plt.plot(X, yy, '+b')
plt.plot(x_grid, yy_true, '-k')
plt.show()
