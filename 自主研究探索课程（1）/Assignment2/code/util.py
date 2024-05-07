import numpy as np
import scipy

def matrix_exp(A):
    # A 为numpy的二维数组，n 为计算步长
    exp_A = scipy.linalg.expm(A)    
    return exp_A

def matrix_power(A, n):
    pow = np.linalg.matrix_power(A, n)
    return pow