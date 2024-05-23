import numpy as np
from matplotlib import pyplot as plt
U2 = [0.043,0.046,0.054,0.071,0.090,0.114,0.138,0.162,0.188,
      0.213,0.240,0.264,0.289,0.313,0.337,0.361,0.384,0.408,
      0.430,0.454,0.477,0.500,0.522,0.545,0.567,0.589]
U1 = [0.034,0.144,0.172,0.189,0.197,0.202,0.204,0.205,0.206,
      0.207,0.207,0.208,0.208,0.210,0.211,0.212,0.215,0.215,
      0.218,0.219,0.221,0.222,0.222,0.224,0.225,0.226]
U2 = np.array(U2)-U2[0]
U1 = np.array(U1)-U1[0]

# U1处理，设置大号字体
plt.xlabel('Time(min)')
plt.ylabel('$U_1$(mV)')
plt.scatter(range(len(U1)), U1, label='$U_1$',color='g')
plt.plot(range(len(U1)), U1, label='$U_1$',color='b',linewidth=1)
plt.axhline(y=0.173, color='r', linestyle='--',linewidth=1,label='0.173mV')
plt.legend()
plt.show()

# U2处理
Uback = U2
t = [i for i in range(3, len(U2))]
U2 = U2[3:]
coefficients = np.polyfit(t, U2, 1)
print(coefficients)

# 使用拟合的参数创建拟合直线
fit_line = np.poly1d(coefficients)

# 计算预测值
line = fit_line([i for i in range(len(Uback))])

# 计算实际值与预测值之间的协方差
cov_xy = np.cov(t, U2, bias=True)[0, 1]
# 计算实际值和预测值的标准差
std_t = np.std(t)
std_U2 = np.std(U2)
r = cov_xy / (std_t * std_U2)

# 打印相关系数,保留10位小数
print(r)

plt.xlabel('Time(min)')
plt.ylabel('$U_2$(mV)')
plt.scatter(range(len(Uback)), Uback, label='$U_2$',color='g')
plt.plot(range(len(Uback)), line, label='Linear Fit of $U_2$',color='b',linewidth=1)
plt.legend()
plt.show()