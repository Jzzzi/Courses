from matplotlib import pyplot as plt
import numpy as np

'''
the main part to plot the figure
'''
# the origin data in list of tuple
# data = [(1.3,1.5),(4.0,5.4),(3.4,7.3),(2.2,3.3),(1.6,4.7),(3.1,0.8)]
data = [(4.51,8),(6.45,6.5),(8.7,5),(7.0,5),(4,0.97),(2.8,6.3)]
# set the chinense fonts and the size of the fonts
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18
# create a figure
plt.figure(figsize=(10, 10), dpi=150) # create a figure
plt.title('拉力-拉出时间关系（水）') # title of the figure
# set the labels of x and y
plt.xlabel('拉出时间（s）')
plt.ylabel('拉力（N）')
# set the range of x and y
plt.xlim(0, 10)
plt.ylim(0, 10)
# scatter points from the data
x, y = zip(*data)
plt.scatter(x, y, color='b', marker='o', label='实测数据')
# plot the line that f = 1/t
x = np.linspace(0.01,10,100)
y = 1/x
比例 = 10.0
plt.plot(x,比例*y, color='r', linestyle='--', linewidth=2,label='反比例关系')
plt.legend()
plt.savefig('水.png') # save the figure
plt.show() # show the figure
plt.close() # exit only when q is pressed
