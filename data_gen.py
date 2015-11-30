from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

# generate datasets
def dataset_fixed_cov( n , dim , seed , C):
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    np.random.seed(seed)
    #C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.ones(dim)]
    y = np.hstack((-1*np.ones(n), np.ones(n)))
    return X, y


def dataset_cov(n , dim , seed):
    '''Generate 2 Gaussians samples with different covariance matrices'''
    np.random.seed(seed)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.ones(dim)]
    y = np.hstack((-1*np.ones(n), np.ones(n)))
    return X, y
