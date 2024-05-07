import numpy as np
from matplotlib import pyplot as plt
from util import *

'''
util.py编写了若干科学计算的函数
命名规则：
小写字母为数值，大写字母为数组（矩阵或张量）
'''
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
deltat = 2.0
m = 100
mg0 = 0.1
tw = lr*qr/(g*hr*a)

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

n = int(np.linalg.norm(AR) * deltat)
dt = deltat/n

ART = AR * dt
Phi = matrix_exp(ART)
print('转移矩阵为')
print(Phi)

Res = np.zeros((m, 5, 1))
Res[:,4,:] = mg0

for i in range(1, m):
    Res[i,:,:] = matrix_power(Phi,n) @ Res[i-1,:,:]

plt.plot([i*deltat for i in range(0, m)], Res[:,0,0], label='x')
plt.plot([i*deltat for i in range(0, m)], Res[:,1,0], label='y')
plt.plot([i*deltat for i in range(0, m)], Res[:,2,0], label='z')
plt.plot([i*deltat for i in range(0, m)], Res[:,3,0], label='h')
plt.legend()
plt.show()