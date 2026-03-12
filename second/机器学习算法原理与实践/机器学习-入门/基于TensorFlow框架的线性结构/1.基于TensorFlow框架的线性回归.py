import tensorflow as tf
import numpy as np


#1.散点输入,定义输入数据
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
#data列表 10*2
# 转换成numpy数组
data = np.array(data)
x_data = data[:,0]
y_data = data[:,1]

x_train = tf.constant(x_data,dtype=tf.float32)
y_train = tf.constant(y_data,dtype=tf.float32)


#2.定义前向模型
#(1,)指得是(None,1)形状
model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=(1,))])
#3.定义损失函数和优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer,loss="mean_squared_error")



#4.开始迭代
epoches = 500

history = model.fit(x_train,y_train,epochs=epoches)
# for epoch in range(1,epoches+1):
#     history= model.fit(x_train,y_train)
#     loss = history.history["loss"][0]
#     print(history.history)
# #5.显示频率设置
#     if epoch % 10 == 0 or epoch == 1:
#         print(f"epoch:{epoch},loss;{loss}")
