import numpy as np
from util import *

# Parameters
ey = 0.76
ex = -1.23
eh = 1.665
eqy = 0.685
eqx = -0.263
eqh = 0.512
eg = 1.0
ty = 0.1 # T^*_y
bp = 0.04
ta = 10.0
qr = 345.0
lr = 600.0
hr = 99.3
a = 69.2
g = 9.8
td = 5.0
bt = 0.8
dt = 2.0
m = 100
mg0 = 0.1
tw = lr*qr/g*hr*a

A = np.zeros((4,4))
A[0,0] = (ex-eg)/ta
A[0,1] = ey/ta
A[0,3] = eh/ta
A[1,0] = -1/ty
A[1,1] = -bp/ty
A[1,2] = -1/ty
A[2,0] = -bt/ty
A[2,1] = -bp*bt/ty
A[2,2] = -(bt/ty+1/td)
A[3,0] = ((eg-ex)*eqx/ta+eqy/ty)/eqh
A[3,1] = (bp*eqy/ty-ey*eqx/ta)/eqh
A[3,2] = eqy/(ty*eqh)
A[3,3] = (1/(tw*eqh)+eh*eqx/(ta*eqh))*(-1)

B = np.zeros((4,1))
B[0,0] = -1/ta
B[3,0] = eqx/(ta*eqh)

AR = np.zeros((5,5))
AR[0:4,0:4] = A
AR[0:4,4] = B.reshape(4)
print('A_R矩阵为')
print(AR)

n = np.linalg.norm(AR) * dt
PAR = matrix_exp(AR)
print('转移矩阵为')
print(matrix_exp(AR))
