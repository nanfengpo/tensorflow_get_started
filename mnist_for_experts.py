# coding:utf-8
# Deep MNIST for Experts
# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

# Start TensorFlow InteractiveSession
sess=tf.InteractiveSession()

x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y=tf.nn.softmax(tf.matmul(x, W)+b)

loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

for i in range(1000):
    batch=mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    if i%50 ==0 :
        print(accuracy.eval(feed_dict={x:mnist.test.images,\
                                       y_:mnist.test.labels}))


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

print " "
