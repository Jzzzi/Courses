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

TD = [1.0, 2.5, 5.0]
BT = [0.6, 0.8, 1.0]
i_max = len(TD)
j_max = len(BT)
# Set the figure resolution
plt.figure(figsize=(15, 15), dpi=150)
# Set the title
plt.suptitle('Different $T_d$ and $b_t$')
for i_td in range(i_max):
    for j_bt in range(j_max):
        td = TD[i_td]
        bt = BT[j_bt]
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
        # print('Matrix A_R is:')
        # print(AR)

        n = int(np.linalg.norm(AR) * deltat)
        dt = deltat/n

        ART = AR * dt
        Phi = matrix_exp(ART)
        # print('The transition matrix is:')
        # print(Phi)

        Res = np.zeros((m, 5, 1))
        Res[:,4,:] = mg0

        for i in range(1, m):
            Res[i,:,:] = matrix_power(Phi,n) @ Res[i-1,:,:]

        plt.subplot(i_max,j_max,i_td*3+j_bt+1)
        # subtitle
        plt.title('$T_d$='+str(td)+', $b_t$='+str(bt))
        # plt.xlabel('Time t')
        # plt.ylabel('Rotational speed n')
        plt.plot([i*deltat for i in range(0, m)], Res[:,0,0], label='Rotational Spennd n')
        # plt.plot([i*deltat for i in range(0, m)], Res[:,1,0], label='y')
        # plt.plot([i*deltat for i in range(0, m)], Res[:,2,0], label='z')
        # plt.plot([i*deltat for i in range(0, m)], Res[:,3,0], label='h')
        # plt.legend()

plt.tight_layout()
plt.savefig('pic/dif_td_bt.png')
# plt.show()
