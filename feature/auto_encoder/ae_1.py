from __future__ import division, print_function, absolute_import
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from prepare.data_split import read_bunch


'''
定义输入层 19
第一层隐含层 8
输出层 19

'''


def ae():
    input_n = 19
    hidden1_n = 9
    hidden2_n = 8
    output_n = 19

    learn_rate = 0.01
    batch_size = 100
    train_epoch = 30000
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(tf.float32, [None, input_n])

    weights1 = tf.Variable(tf.random.normal([input_n, hidden1_n], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[hidden1_n]))

    weights2 = tf.Variable(tf.random.normal([hidden1_n, hidden2_n], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[hidden2_n]))

    weights3 = tf.Variable(tf.random.normal([hidden2_n, output_n], stddev=0.1))
    bias3 = tf.Variable(tf.constant(0.1, shape=[output_n]))
    a1 = tf.nn.sigmoid(tf.matmul(x, weights1) + bias1)
    a2 = tf.nn.sigmoid(tf.matmul(a1, weights2) + bias2)
    y_ = tf.nn.sigmoid(tf.matmul(a2, weights3) + bias3)

    loss = tf.reduce_mean(tf.pow(y_ - x, 2))
    train_op = tf.compat.v1.train.RMSPropOptimizer(learn_rate).minimize(loss)
    _, m = x_train.shape
    n, d = x_train.shape
    num_batches = n // batch_size
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        for i in range(train_epoch):
            for j in range(num_batches):
                batch = x_train[j * batch_size: (j * batch_size + batch_size)]
                _, ob = sess.run([train_op, loss], feed_dict={x: batch})
                if j % 100 == 0 and i % 100 == 0:
                    print('trainint epoch {0} batch {2} cost {1}'.format(i, ob, j))
                    print(weights1)


if __name__ == '__main__':
    data_set = read_bunch('e:/Paper/imb_problem/data/data_set.data')
    trX, teX, trY, teY = data_set.X_train, data_set.X_test, data_set.y_train, data_set.y_test

    x_train = trX.astype(np.float32)
    x_test = teX.astype(np.float32)
    ae()


