import numpy as np
from matplotlib import pyplot as plt
from util import *
import tkinter as tk

'''
util.py编写了若干科学计算的函数
命名规则：
小写字母为数值，大写字母为数组（矩阵或张量）
'''

import tkinter as tk

# Set the main figure
root = tk.Tk()
root.title("Transient Parameters")
root.geometry("400x600")
root.geometry("+100+200")

# Define the six parameters
eys = tk.DoubleVar(value=0.76)
exs = tk.DoubleVar(value=-1.23)
ehs = tk.DoubleVar(value=1.665)
eqys = tk.DoubleVar(value=0.685)
eqxs = tk.DoubleVar(value=-0.263)
eqhs = tk.DoubleVar(value=0.512)

# Create sliders to set the six parameters
slider_values = [eys, exs, ehs, eqys, eqxs, eqhs]
slider_labels = ["ey", "ex", "eh", "eqy", "eqx", "eqh"]
for i, (value, label) in enumerate(zip(slider_values, slider_labels)):
    slider = tk.Scale(root, from_=-4.0, to=4.0, resolution=0.01, orient=tk.HORIZONTAL, label=label, variable=value)
    slider.pack(fill=tk.Y,pady=10)

# The function to calculate the result
def calculate():
    # Close the previous plot figure
    plt.close()
    # Parameters
    ey = eys.get()
    ex = exs.get()
    eh = ehs.get()
    eqy = eqys.get()
    eqx = eqxs.get()
    eqh = eqhs.get()

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

    # The system is stable if all eigenvalues of A have negative real parts
    eigvals, eigvecs = eig(A)
    print(f'The eigenvalues of A are:')
    print(eigvals)

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

    plt.figure()
    # Set the position of the figure on the screen
    plt.get_current_fig_manager().window.wm_geometry("+600+200")
    plt.title('Rotational speed n with time')
    plt.xlabel('Time t')
    plt.ylabel('Rotational speed n')
    plt.plot([i*deltat for i in range(0, m)], Res[:,0,0], label='Rotational Spennd n')
    # plt.plot([i*deltat for i in range(0, m)], Res[:,1,0], label='y')
    # plt.plot([i*deltat for i in range(0, m)], Res[:,2,0], label='z')
    # plt.plot([i*deltat for i in range(0, m)], Res[:,3,0], label='h')
    plt.legend()
    # plt.show()
    # plt.savefig('pic/n-t.png')
    plt.show()
    # Exit the plot figure when the key is pressed

# Create a button to calculate the result
button = tk.Button(root, text="Calculate the result", command=calculate)
button.pack(fill=tk.Y,pady=10)

def resetslider():
    eys.set(0.76)
    exs.set(-1.23)
    ehs.set(1.665)
    eqys.set(0.685)
    eqxs.set(-0.263)
    eqhs.set(0.512)

reset = tk.Button(root, text="Reset the parameters", command=resetslider)
reset.pack(fill=tk.Y,pady=10)

# The main loop
root.mainloop()