import numpy as np
import scipy

def matrix_exp(A):
    '''
    Compute the matrix exponential of A
    '''
    exp_A = scipy.linalg.expm(A)    
    return exp_A

def matrix_power(A, n):
    '''
    Compute the n-th power of A
    '''
    pow = np.linalg.matrix_power(A, n)
    return pow