import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(5,4))
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['figure.dpi'] = 1000
# 画图
x = np.array([1,2,3,4,5,6,7,8])
y1 = np.array([28,26,33,35,24,18,4,1])
y2 = np.array([26,25,31,33.4,23.6,15.2,2.2,0.4])
z1 = np.array([5.3,8.2,20.2,17.3,26,27.1,32.8,35])
z2 = np.array([4.6,7.8,18.7,15.8,23.4,26.1,30.2,34.4])
dashes = [10, 3, 100, 3]

l1= plt.plot(x, y1,"-ok", label='谣言真实值')
l2 = plt.plot(x, y2,"-or", label='谣言预测值')
l3= plt.plot(x, z1,"-ob", label='辟谣真实值')
l4 = plt.plot(x, z2,"-om", label='辟谣预测值')
plt.axis([1,8,0,35])
plt.xlabel('时间切片',fontsize=16)
plt.ylabel('转发人数',fontsize=16)
plt.legend(loc='lowerright')
plt.title('整体传播预测',fontsize=18)
fig=plt.gcf()
fig.set_facecolor("#F0FAFF")
ax1=plt.gca()
ax1.patch.set_facecolor("#F0FAFF")
ax1.patch.set_alpha(0.5)
plt.show()

