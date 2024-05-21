import numpy as np
import matplotlib.pyplot as plt

H_R = 150
Q_0 =0.4774321
H_0 = 143.48828
#Question1计算结果
L = 600
a = 1200
D = 0.5
f = 0.018
# k is a coefficient
k = 0.009
g = 9.806
t_c = 0.5
E_m = 1.5
rho = 1000
# The minimum water head at which the water will vaporize
H_min = -1.013e5/g/rho
# The number of nodes
N = 20
# The parameters of the iteration
dx = L/N
dt = dx/a
T = 25
N_t = int(T/dt)

# initialize the Q, H and E, E is the volumn of the cavity, axis 0 is time, axis 1 is the position
init = np.zeros(N+1)
Q = np.zeros((N_t+1,N+1))
H = np.zeros((N_t+1,N+1)) + H_R
E = np.zeros((N_t+1,N+1))
# initialize the H by linear interpolation
for i in range(N+1):
    H[0,i] = H_R+(H_0-H_R)/(N)*i

A = np.pi*D**2/4
B = a/(g*A)
S = 0
R = f*dx/(2*g*D*A**2)

for i in range(1,N_t+1):
    t =  i*dt
    if t < t_c:
        tau = t/t_c
    else:
        tau = 1
    for j in range(N+1):
        # If the cavitarion does not happen
        if E[i-1,j] <= 0:
            # The boundary condition at the start of the pipe
            if j == 0:
                H[i,j] = H_R
                C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
                Q[i,j] = (H[i,j]-C_M)/B
            # The boundary condition at the end of the pipe
            elif j == N:
                C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                C_V = (Q_0*tau)**2/(2*H_0)
                Q[i,j] = -B*C_V+np.sqrt((B*C_V)**2+2*C_V*C_P)
                H[i,j] = C_P - B*Q[i,j]
            # The iteration of the middle nodes
            else:
                C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
                H[i,j] = (C_P+C_M)/2
                Q[i,j] = (C_P-C_M)/(2*B)
        # If the cavitation happens
        else:
            pass

# Plot the relationship between the water head and the time of all the nodes
# set the resolution and the dpi of the plot
plt.figure(figsize=(12,10), dpi=100)

for i in range(N+1):
    if i%10 == 0:
        plt.plot(np.arange(0,N_t+1)*dt,H[:,i],label=f'node N={i}')
# Draw line H_min, and the maximum water head
plt.plot(np.arange(0,N_t+1)*dt,np.ones(N_t+1)*H_min,label=f'H_min={H_min}',linestyle='--')
plt.plot(np.arange(0,N_t+1)*dt,np.ones(N_t+1)*np.max(H[:,N]),label=f'H_max={np.max(H[:,N])}',linestyle='--') 
plt.xlabel('Time (s)')
plt.ylabel('Water head (m)')
plt.title('The water head in the pipe at different time and position')
plt.legend()
plt.savefig('./Report_2/pic/MOC_Open_Waterhead.png')
# close the plot
plt.close()
# Plot the relationship between the flow rate and the time of all the nodes
# set the resolution and the dpi of the plot
plt.figure(figsize=(10,10), dpi=100)
for i in range(N+1):
    if i%10 == 0:
        plt.plot(np.arange(0,N_t+1)*dt,Q[:,i],label=f'node N={i}')
plt.xlabel('Time (s)')
plt.ylabel('Flow rate (m^3/s)')
plt.title('The flow rate in the pipe at different time and position')
plt.legend()
plt.savefig('./Report_2/pic/MOC_Open_Flowrate.png')
plt.close()
