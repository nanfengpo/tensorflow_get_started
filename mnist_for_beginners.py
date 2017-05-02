# coding:utf-8
# MNIST For ML Beginners：：使用全连接神经网络模型解决MNIST问题
# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)


x=tf.placeholder(tf.float32, [None, 784])
W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))

# 使用softmax回归模型预测
y=tf.nn.softmax(tf.matmul(x, W)+b)
# 添加一个占位符用于输入正确值
y_=tf.placeholder("float",[None,10])

# 计算交叉熵
loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化变量
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_xs,y_:batch_ys})
        # 评估模型
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
        if i%50 == 0:
            print(sess.run(accuracy,feed_dict=\
                {x:mnist.test.images,y_:mnist.test.labels}))

