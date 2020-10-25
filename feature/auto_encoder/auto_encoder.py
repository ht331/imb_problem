"""
    自编码器
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import bunch


class AutoEncoder(object):

    def __init__(self, m, n, act_fun,  eta=0.01):
        """

        :param m:   number of neurons in input/output layer
        :param n:  number of neurons in hidden layer
        :param eta:
        """

        self._m = m
        self._n = n
        self.act_func = act_fun
        self.learning_rate = eta
        tf.compat.v1.disable_eager_execution()

        # weighs and biases

        self._w1 = tf.Variable(tf.random.normal(shape=(self._m, self._n)))
        self._w2 = tf.Variable(tf.random.normal(shape=(self._n, self._m)))
        self._b1 = tf.Variable(np.zeros(self._n).astype(np.float32))
        self._b2 = tf.Variable(np.zeros(self._m).astype(np.float32))

        # 输入的占位符
        self._x = tf.compat.v1.placeholder(tf.float32, [None, self._m])
        self.y = self.encoder(self._x)
        self.r = self.decoder(self.y)

        error = self._x - self.r
        self._loss = tf.reduce_mean(tf.pow(error, 2))
        self._opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self._loss)

    def encoder(self, x):
        h = tf.matmul(x, self._w1) + self._b1
        if self.act_func == 'sigmoid':
            return tf.nn.sigmoid(h)
        elif self.act_func == 'tanh':
            return tf.nn.tanh(h)
        elif self.act_func == 'Relu':
            return tf.nn.relu(h)
        elif self.act_func == 'elu':
            return tf.nn.elu(h)
        elif self.act_func == 'selu':
            return tf.nn.selu(h)
        elif self.act_func == 'swish':
            return h * tf.nn.sigmoid(h)
        elif self.act_func == 'leakyrelu':
            return tf.nn.leaky_relu(h)

    def decoder(self, x):
        h = tf.matmul(x, self._w2) + self._b2
        if self.act_func == 'sigmoid':
            return tf.nn.sigmoid(h)
        elif self.act_func == 'tanh':
            return tf.nn.tanh(h)
        elif self.act_func == 'Relu':
            return tf.nn.relu(h)
        elif self.act_func == 'elu':
            return tf.nn.elu(h)
        elif self.act_func == 'selu':
            return tf.nn.selu(h)
        elif self.act_func == 'swish':
            return h * tf.nn.sigmoid(h)
        elif self.act_func == 'leakyrelu':
            return tf.nn.leaky_relu(h)

    def set_session(self, session):
        self.session = session

    def reduced_dimension(self, x):
        h = self.encoder(x)
        return self.session.run(h, feed_dict={self._x: x})

    def reconstruct(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return self.session.run(r, feed_dict={self._x: x})

    def fit(self, x, epochs=10, batch_size=1000):
        n, d = x.shape
        num_batches = n // batch_size

        obj = []
        for i in range(epochs):
            for j in range(num_batches):
                batch = x[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self.session.run([self._opt, self._loss], feed_dict={self._x: batch})
                if j % 100 == 0 and i % 1000 == 0:
                    print('trainint epoch {0} batch {2} cost {1}'.format(i, ob, j))
                obj.append(ob)
        return obj


if __name__ == '__main__':
    from prepare.data_split import read_bunch, write_bunch
    import bunch
    # data_set = read_bunch('e:/Paper/imb_problem/data/data_set.data')
    data_set = read_bunch('e:/Paper/imb_problem/data/creditcard.data')

    trX, teX, trY, teY = data_set.X_train, data_set.X_test, data_set.y_train, data_set.y_test
    x_train = trX.astype(np.float32)
    x_test = teX.astype(np.float32)
    n, m = x_train.shape
    hidden = 15
    epo = 300
    act_fun_list = ['sigmoid', 'tanh', 'elu', 'selu', 'swish']
    for af in act_fun_list:
        ae = AutoEncoder(m, hidden, af)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            ae.set_session(sess)
            err = ae.fit(x_train, epochs=epo)
            daf_train = ae.reduced_dimension(x_train)
            daf_test = ae.reduced_dimension(x_test)

        dataset = bunch.Bunch(X_train=daf_train,
                              y_train=trY,
                              X_test=daf_test,
                              y_test=teY)

        write_bunch('e:/Paper/imb_problem/data/creditcard_%s_%s.data' % (af, hidden), dataset)

