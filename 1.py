
# coding:utf-8
'''
【目的】
    介绍Tensorflow的基本使用
    • 使用图 (graph) 来表示计算任务.
	• 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
	• 使用 tensor 表示数据.
	• 通过 变量 (Variable) 维护状态.
	• 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

'''

import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1=tf.constant([[3.,3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2=tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product=tf.matmul(matrix1,matrix2)

# 启动默认图.
sess=tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result=sess.run(product)
print(result)

# 任务完成, 关闭会话.
sess.close()


# 创建一个变量, 初始化为标量 0.
state=tf.Variable(0,name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one=tf.constant(1)
new_value= tf.add(state,one)# 其实，直接写成new_value=state+2也可以！！！
update=tf.assign(state,new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op=tf.global_variables_initializer()

# 启动图, 运行 op
with tf.Session() as sess:
    sess.run(init_op) # 初始化变量
    print(sess.run(state))
    for i in range(3):
        sess.run(update)
        print(sess.run(state))


# Feed
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))


