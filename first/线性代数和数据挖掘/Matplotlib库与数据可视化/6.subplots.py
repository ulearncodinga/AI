'''
在Matplotlib中，subplots用于创建包含多个子图（subplot）的图形。这允许你在
同一个 Figure 对象内组织多个图表，从而方便地比较数据或者展示多方面的信息。
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,0.01)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = 1/(1+np.exp(-x)) #Sigmoid function

#创建一个2x2的子图网格
fig,axs = plt.subplots(2,2)
#在第一个子图中写入数据
axs[0,0].plot(x,y1)
axs[0,0].set_title("Since")
axs[0,0].set_xlabel('X')
axs[0,0].set_ylabel('Y')
#第二个子图写入数据
axs[0,1].plot(x,y2)
axs[0,1].set_title("Cos")
axs[0,1].set_xlabel('X')
axs[0,1].set_ylabel('Y')
#第三个子图写入数据
axs[1,0].plot(x,y3)
axs[1,0].set_title("Tan")
axs[1,0].set_xlabel('X')
axs[1,0].set_ylabel('Y')
#第四个子图写入数据
axs[1,1].plot(x,y4)
axs[1,1].set_title("Sigmoid")
axs[1,1].set_xlabel('X')
axs[1,1].set_ylabel('Y')






#调整子图的间距
plt.tight_layout()


plt.show()
