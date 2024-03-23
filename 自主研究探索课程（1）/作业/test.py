import numpy as np
H_R = 150
N = 10
init = np.zeros(N+1)
Q = np.array([init])
H = np.array([init])+H_R
print(Q.shape,H.shape)
Q = np.append(Q,[init],axis=0)
H = np.append(H,[init],axis=0)
print(Q,H)
print(Q.shape,H.shape)