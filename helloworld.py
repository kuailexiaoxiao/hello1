import tensorflow as tf
import random
import numpy as np
sess = tf.Session()
input_data = tf.Variable(np.random.rand(10,9,9,3),dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2,3,3,2),dtype=np.float32)
y = tf.nn.depthwise_conv2d(input_data,filter_data,strides=[1,1,1,1],padding='SAME')
print sess.run(tf.shape(y))

