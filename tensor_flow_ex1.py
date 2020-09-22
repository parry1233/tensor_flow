import tensorflow as tf
import numpy as np

# create test data
X_data = np.random.rand(500).astype(np.float32)
Y_data = X_data*0.5 + 0.3

# create tensorflow structure start###

#create by arbitrary variable 比重 = 隨機數([1維],範圍初始從-2.0,至2.0)
Weights = tf.Variable(tf.random_uniform([1],-2.0,2.0))
#create variable 偏差 = 初始為零([1維])
biases = tf.Variable(tf.zero([1]))

y= Weights*X_data + biases
#計算 y 與 Y_data 實際上的差異
loss = tf.reduce_mean(tf.square(y-Y_data))
#已知 y 與 Y_data會有差異，建立優化器減少誤差 GradientDescentOptimizer(學習效率)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train= optimizer.minimize(loss)

init = tf.initialize_all_variables()

# create tensorflow structure end###

# tf.Session()為一個指針，指向要處理的地方
session = tf.Session()
# 啟動指針
session.run(init)

for step in range(201):
    session.run(train)
    #每隔 20step 就輸出 Weights 與 biases
    if step % 20 == 0:
        print(step,session.run(Weights),session.run(biases))