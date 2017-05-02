# Deep MNIST for Experts：使用卷积神经网络模型解决MNIST问题
# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

# Start TensorFlow InteractiveSession

x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])

# to Build a Multilayer Convolutional Network, create weights and biases
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],\
                          strides=[1,2,2,1],padding='SAME')

# 第1卷积层C1：32@28*28
# 32个特征图，每个特征图是输入（28*28）与卷积核（5*5）做卷积得到的结果
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
x_image=tf.reshape(x,[-1,28,28,1])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

# 第1池化层C1：32@14*14
# Max-Pooling: 选择2*2的Pooling窗口中的最大值作为采样值
h_pool1=max_pool_2x2(h_conv1)

# 第2卷积层C1：64@14*14
# 32个特征图，每个特征图是输入（28*28）与卷积核（5*5）做卷积得到的结果
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

# 第2池化层C1：64@7*7
# Max-Pooling: 选择2*2的Pooling窗口中的最大值作为采样值
h_pool2=max_pool_2x2(h_conv2)

# 全连接层1(full connection)：64*7*7->1024
W_fc1=weight_variable([64*7*7,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,64*7*7])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# Dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# 输出层(全连接层2):1024->10
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 训练过程
loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    # 设置TensorBoard
    tf.summary.scalar("accuracy",accuracy)
    merged=tf.summary.merge_all()
    tb_writer=tf.summary.FileWriter("TensorBoard/",graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch=mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
        if i%100 ==0:
            # train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            # print("step %d, training accuracy %g" %(i,train_accuracy))
            result=sess.run(merged,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            tb_writer.add_summary(result,i)
    print("test accuracy %g" % sess.run(accuracy,feed_dict={x:batch[0],\
                y_:batch[1],keep_prob:1.0}))



