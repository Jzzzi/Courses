'''
neglect the effect of the slope of the pipe
'''
import numpy as np
import matplotlib.pyplot as plt

H_R = 32.0 # the reservoir head
# Results in Question 1
L = 37.23 # the length of the pipe
a = 1319.0 # the speed of wave
D = 0.0221 # the diameter of the pipe
A = 1/4*np.pi*D**2 # the cross section area of the pipe
f = 0.018 # the friction factor
k = 0.009 # k is a coefficient of the valve
g = 9.806 # the gravity
t_c = 0.009 # the closure time of the valve
rho = 1000 # the density of the water
H_min = -1.013e5/g/rho # the minimum water head at which the water will vaporize
N = 16 # the number of nodes
V_0 = 1.5 # the initial velocity of the water
Q_0 = A*V_0 # the initial flow rate
H_0 = 32.0 # the initial water head
# The parameters of the iteration
dx = L/N
dt = dx/a
print(f'dt={dt}')
T = 25
psi = 1.0 # the weighting factor
N_t = int(T/dt)

# initialize the Q, H and E, E is the volumn of the cavity, axis 0 is time, axis 1 is the position
# Up stream flow rate
Qu = np.zeros((N_t+1,N+1))
Qu[0,:] = Q_0
# Down stream flow rate
Q = np.zeros((N_t+1,N+1))
Q[0,:] = Q_0
H = np.zeros((N_t+1,N+1))
H[0,:] = H_0
E = np.zeros((N_t+1,N+1))

A = np.pi*D**2/4
B = a/(g*A)
S = 0
# R = f*dx/(2*g*D*A**2)
R = 0

for i in range(1,N_t+1):
    t =  i * dt
    if t < t_c:
        tau = 1 - t/t_c
    else:
        tau = 0
    for j in range(N+1):
        # If the cavitation does not happen, E[i-1,j] <= 0
        if E[i-1,j] <= 0:
            if j == 0:
                H[i,j] = H_R
                C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
                Q[i,j] = (H[i,j]-C_M)/B
                Qu[i,j] = Q[i,j]
            if j > 0 and j < N:
                C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
                H[i,j] = (C_P+C_M)/2
                Q[i,j] = (C_P-C_M)/(2*B)
                Qu[i,j] = Q[i,j]
                if H[i,j] < H_min:
                    H[i,j] = H_min
                    Qu[i,j] = (-H[i,j]+H[i-1,j-1]+B*Q[i-1,j-1])/(B+R*abs(Q[i-1,j-1]))
                    Q[i,j] = (H[i,j]-H[i-1,j+1]+B*Qu[i-1,j+1])/(B+R*abs(Qu[i-1,j+1]))
                    E[i,j] = E[i-1,j] + psi*(Q[i,j]*dt - Qu[i,j]*dt) + (1-psi)*(Q[i-1,j]*dt - Qu[i-1,j]*dt)
                    # E[i,j] = E[i-2,j] + psi*(Q[i,j]*2*dt - Qu[i,j]*2*dt) + (1-psi)*(Q[i-2,j]*2*dt - Qu[i-2,j]*2*dt)
            if j == N:
                C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                C_V = (Q_0*tau)**2/(2*H_0)
                Q[i,j] = -B*C_V+np.sqrt((B*C_V)**2+2*C_V*C_P)
                H[i,j] = C_P - B*Q[i,j]
                Qu[i,j] = Q[i,j]
                if H[i,j] < H_min:
                    if t < t_c:
                        raise("Cavitation before t_c")
                    H[i,j] = H_min
                    Qu[i,j] = (-H[i,j]+H[i-1,j-1]+B*Q[i-1,j-1])/(B+R*abs(Q[i-1,j-1]))
                    Q[i,j] = 0
                    E[i,j] = E[i-1,j] + psi*(Q[i,j]*dt - Qu[i,j]*dt) + (1-psi)*(Q[i-1,j]*dt - Qu[i-1,j]*dt)
                    # E[i,j] = E[i-2,j] + psi*(Q[i,j]*2*dt - Qu[i,j]*2*dt) + (1-psi)*(Q[i-2,j]*2*dt - Qu[i-2,j]*2*dt)
        # If the cavitation happens, E[i,j] > 0
        elif E[i-1,j] > 0:
            if j == 0:
                raise("Cavatation at the start")
            if j > 0 and j < N:
                H[i,j] = H_min
                Qu[i,j] = (-H[i,j]+H[i-1,j-1]+B*Q[i-1,j-1])/(B+R*abs(Q[i-1,j-1]))
                Q[i,j] = (H[i,j]-H[i-1,j+1]+B*Qu[i-1,j+1])/(B+R*abs(Qu[i-1,j+1]))
                # E[i,j] = E[i-1,j] + psi*(Q[i,j]*dt - Qu[i,j]*dt) + (1-psi)*(Q[i-1,j]*dt - Qu[i-1,j]*dt)
                E[i,j] = E[i-2,j] + psi*(Q[i,j]*2*dt - Qu[i,j]*2*dt) + (1-psi)*(Q[i-2,j]*2*dt - Qu[i-2,j]*2*dt)
                if E[i, j] < 0:
                    E[i, j] = 0
                    C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                    C_M = H[i-1,j+1]-Q[i-1,j+1]*(B-S-R*abs(Q[i-1,j+1]))
                    H[i,j] = (C_P+C_M)/2
                    Q[i,j] = (C_P-C_M)/(2*B)
                    Qu[i,j] = Q[i,j]
                    if H[i,j] < H_min:
                        H[i,j] = H_min
            if j == N:
                H[i,j] = H_min
                Qu[i,j] = (-H[i,j]+H[i-1,j-1]+B*Q[i-1,j-1])/(B+R*abs(Q[i-1,j-1]))
                C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                C_V = (Q_0*tau)**2/(2*H_0)
                Q[i,j] = -B*C_V+np.sqrt((B*C_V)**2+2*C_V*C_P)
                # E[i,j] = E[i-1,j] + psi*(Q[i,j]*dt - Qu[i,j]*dt) + (1-psi)*(Q[i-1,j]*dt - Qu[i-1,j]*dt)
                E[i,j] = E[i-2,j] + psi*(Q[i,j]*2*dt - Qu[i,j]*2*dt) + (1-psi)*(Q[i-2,j]*2*dt - Qu[i-2,j]*2*dt)
                if E[i,j] < 0:
                    E[i,j] = 0
                    C_P = H[i-1,j-1]+Q[i-1,j-1]*(B+S-R*abs(Q[i-1,j-1]))
                    C_V = (Q_0*tau)**2/(2*H_0)
                    Q[i,j] = -B*C_V+np.sqrt((B*C_V)**2+2*C_V*C_P)
                    H[i,j] = C_P - B*Q[i,j]
                    Qu[i,j] = Q[i,j]
                    if H[i,j] < H_min:
                        H[i,j] = H_min




#===================Plot Part=======================#


# Plot the relationship between the water head and the time of all the nodes
# set the resolution and the dpi of the plot
plt.figure(figsize=(12,10), dpi=100)

for i in range(N+1):
    # if i%int(int(N)/2) == 0:
    if i == int(N/2):
    # if i == N:
        plt.plot(np.arange(0,N_t+1)*dt,H[:,i],label=f'node N={i}')
# Draw line H_min, and the maximum water head
plt.plot(np.arange(0,N_t+1)*dt,np.ones(N_t+1)*H_min,label=f'H_min={H_min}',linestyle='--')
plt.plot(np.arange(0,N_t+1)*dt,np.ones(N_t+1)*np.max(H[:,N]),label=f'H_max={np.max(H[:,N])}',linestyle='--') 
plt.xlabel('Time (s)')
plt.ylabel('Water head (m)')
plt.title('The water head in the pipe at different time and position')
plt.legend()
plt.xlim(0,0.80)
plt.savefig('./pic/DVCM_Waterhead_new.png')
# close the plot
plt.close()

# Plot the relationship between the flow rate and the time of all the nodes
plt.figure(figsize=(12,10), dpi=100)
for i in range(N+1):
    # if i%int(int(N)/2) == 0:
    # if i == int(N/2):
    if i == N:
        plt.plot(np.arange(0,N_t+1)*dt,Q[:,i],label=f'node N={i}')
plt.xlabel('Time (s)')
plt.ylabel('Flow rate (m^3/s)')
plt.title('The flow rate in the pipe at different time and position')
plt.xlim(0,0.80)
plt.legend()
plt.savefig('./pic/DVCM_Flowrate_new.png')
plt.close()

# Plot the bubble volumn
plt.figure(figsize=(10,10), dpi=100)
for i in range(0, N+1):
    if i % int(N/5) == 0:
        plt.plot(np.arange(0,N_t+1)*dt,E[:,i],label=f'node N={i}')
plt.xlabel('Time (s)')
plt.ylabel('Volumn m^3')
plt.title('Bubble Volumn')
# plt.xlim(0,0.80)
plt.legend()
plt.savefig('./pic/DVCM_Bubble_new.png')
# plt.show()
plt.close()