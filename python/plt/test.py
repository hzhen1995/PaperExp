import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(5,4))
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['figure.dpi'] = 1000
# 画图
x = np.array([1,2,3,4,5,6,7,8])
y1 = np.array([28,26,33,35,24,18,4,1])
y2 = np.array([26,24.6,29.2,30.4,23.6,10.2,2.2,0.4])
y3 = np.array([22.3,23.2,27.4,19.6,16,3.5,1.1,0.3])
dashes = [10, 3, 100, 3]

l1= plt.plot(x, y1,"-ok", label='真实值')
l2 = plt.plot(x, y2,"-or", label='全部特征预测')
l3 = plt.plot(x, y3,"-ob", label='内部特征预测')
plt.axis([1,8,0,35])
plt.xlabel('时间切片',fontsize=16)
plt.ylabel('转发人数',fontsize=16)
plt.legend(loc='lowerright')
plt.title('谣言传播预测',fontsize=18)
fig=plt.gcf()
fig.set_facecolor("#F0FAFF")
ax1=plt.gca()
ax1.patch.set_facecolor("#F0FAFF")
ax1.patch.set_alpha(0.5)
plt.show()

