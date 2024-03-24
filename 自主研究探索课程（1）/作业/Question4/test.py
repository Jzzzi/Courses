import numpy as np
import matplotlib.pyplot as plt

H_R = 150
L = 600
a = 1200
D = 0.5
f = 0.018
k = 0.009
# 即C_dA_g
g = 9.806
t_c = 4.0
#阀门关闭时间
t_s = 2.0
#阀门开启时间
E_m = 1
Q_0 = k*np.sqrt(2*g*H_R)
# 阀门全开时的流量

N = 10
# 管道分段

# 初始化Q和H
init = np.zeros(N+1)
Q = np.array([init])
H = np.array([init])+H_R
# H[0,N] = 0

# 迭代计算
dx = L/N
dt = dx/a

N_t = int(35/dt)#这里为仿真时间
A = np.pi*D**2/4
B = a/(g*A)
S = 0
R = f*dx/(2*g*D*A**2)
for i in range(1,N_t+1):
    t =  i*dt
    if t<t_s:
        tau = 0
        # tau = t/t_s
        # tau = np.power(1-t/t_c,E_m)
        # tau = 1
    else:
        # tau = 1
        tau = 0
    Q = np.append(Q,[init],axis=0)
    H = np.append(H,[init],axis=0)
    for j in range(N+1):
        if j == 0:
            H[i,j] = H_R
            C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
            Q[i,j] = (H[i,j]-C_M)/B
        elif j == N:
            C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
            C_V = (Q_0*tau)**2/(2*H_R)
            Q[i,j] = -B*C_V+np.sqrt((B*C_V)**2+2*C_V*C_P)
            H[i,j] = C_P - B*Q[i,j]
        else:
            C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
            C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
            H[i,j] = (C_P+C_M)/2
            Q[i,j] = (C_P-C_M)/(2*B)

# 画出流量Q随时间变化
fig, ax = plt.subplots(1,3)
fig.set_size_inches(12,4)
plt.rcParams['font.family'] = 'Times New Roman'


fig.suptitle('NodeTraffic($t_s$='+str(t_s)[0:4]+')')
ax[0].plot([i*dt for i in range (N_t+1)],Q[:,0])
ax[0].set_title('Node 0 Traffic')
ax[1].plot([i*dt for i in range (N_t+1)],Q[:,int(N/2)])
ax[1].set_title('Node '+str(int(N/2))+' Traffic')
ax[2].plot([i*dt for i in range (N_t+1)],Q[:,N])
ax[2].set_title('Node '+str(N)+' Traffic')
plt.tight_layout()
# 存储图像到Question4文件夹下
plt.show()

# 画出水头H随时间变化
fig, ax = plt.subplots(1,3)
fig.set_size_inches(12,4)
fig.suptitle('NodeHead($t_s$='+str(t_s)[0:4]+')')
ax[0].plot([i*dt for i in range (N_t+1)],H[:,0])
ax[0].set_title('Node 0 Head')
ax[1].plot([i*dt for i in range (N_t+1)],H[:,int(N/2)])
ax[1].set_title('Node '+str(int(N/2))+' Head')
ax[2].plot([i*dt for i in range (N_t+1)],H[:,N])
ax[2].set_title('Node '+str(N)+' Head')
plt.tight_layout()
plt.show()