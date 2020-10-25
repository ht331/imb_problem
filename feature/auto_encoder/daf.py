import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(object):

    def __init__(self, m, h1, h2, h3, h4, act_func, eta=0.01):
        """

        :param m:   number of neurons in input/output layer
        :param n:  number of neurons in hidden layer
        :param eta:
        """

        self._m = m
        self._n1 = h1
        self._n2 = h2
        self._n3 = h3
        self._n4 = h4
        self.act_func = act_func
        self.learning_rate = eta
        tf.compat.v1.disable_eager_execution()

        # weighs and biases

        self._w1 = tf.Variable(tf.random.normal(shape=(self._m, self._n1)))
        self._w2 = tf.Variable(tf.random.normal(shape=(self._n1, self._n2)))
        self._w3 = tf.Variable(tf.random.normal(shape=(self._n2, self._n3)))
        self._w4 = tf.Variable(tf.random.normal(shape=(self._n3, self._n4)))
        self._w5 = tf.Variable(tf.random.normal(shape=(self._n4, self._m)))

        self._b1 = tf.Variable(np.zeros(self._n1).astype(np.float32))
        self._b2 = tf.Variable(np.zeros(self._n2).astype(np.float32))
        self._b3 = tf.Variable(np.zeros(self._n3).astype(np.float32))
        self._b4 = tf.Variable(np.zeros(self._n4).astype(np.float32))
        self._b5 = tf.Variable(np.zeros(self._m).astype(np.float32))
        # 输入的占位符
        self._x = tf.compat.v1.placeholder(tf.float32, [None, self._m])
        self.y = self.encoder(self._x)
        self.r = self.decoder(self.y)

        error = self._x - self.r
        self._loss = tf.reduce_mean(tf.pow(error, 2))
        self._opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self._loss)

    def encoder(self, x):
        if self.act_func == 'sigmoid':
            a1 = tf.nn.sigmoid(tf.matmul(x, self._w1) + self._b1)
            a2 = tf.nn.sigmoid(tf.matmul(a1, self._w2) + self._b2)
            a3 = tf.nn.sigmoid(tf.matmul(a2, self._w3) + self._b3)
            a4 = tf.nn.sigmoid(tf.matmul(a3, self._w4) + self._b4)
            return a4
        elif self.act_func == 'tanh':
            a1 = tf.nn.tanh(tf.matmul(x, self._w1) + self._b1)
            a2 = tf.nn.tanh(tf.matmul(a1, self._w2) + self._b2)
            a3 = tf.nn.tanh(tf.matmul(a2, self._w3) + self._b3)
            a4 = tf.nn.tanh(tf.matmul(a3, self._w4) + self._b4)
            return a4
        elif self.act_func == 'relu':
            a1 = tf.nn.relu(tf.matmul(x, self._w1) + self._b1)
            a2 = tf.nn.relu(tf.matmul(a1, self._w2) + self._b2)
            a3 = tf.nn.relu(tf.matmul(a2, self._w3) + self._b3)
            a4 = tf.nn.relu(tf.matmul(a3, self._w4) + self._b4)
            return a4
        elif self.act_func == 'elu':
            a1 = tf.nn.elu(tf.matmul(x, self._w1) + self._b1)
            a2 = tf.nn.elu(tf.matmul(a1, self._w2) + self._b2)
            a3 = tf.nn.elu(tf.matmul(a2, self._w3) + self._b3)
            a4 = tf.nn.elu(tf.matmul(a3, self._w4) + self._b4)
            return a4
        elif self.act_func == 'selu':
            a1 = tf.nn.selu(tf.matmul(x, self._w1) + self._b1)
            a2 = tf.nn.selu(tf.matmul(a1, self._w2) + self._b2)
            a3 = tf.nn.selu(tf.matmul(a2, self._w3) + self._b3)
            a4 = tf.nn.selu(tf.matmul(a3, self._w4) + self._b4)
            return a4
        elif self.act_func == 'leakyrelu':
            a1 = tf.nn.leaky_relu(tf.matmul(x, self._w1) + self._b1)
            a2 = tf.nn.leaky_relu(tf.matmul(a1, self._w2) + self._b2)
            a3 = tf.nn.leaky_relu(tf.matmul(a2, self._w3) + self._b3)
            a4 = tf.nn.leaky_relu(tf.matmul(a3, self._w4) + self._b4)
            return a4

    def decoder(self, x):
        if self.act_func == 'sigmoid':
            h = tf.nn.sigmoid(tf.matmul(x, self._w5) + self._b5)
            return h
        elif self.act_func == 'tanh':
            h = tf.nn.tanh(tf.matmul(x, self._w5) + self._b5)
            return h
        elif self.act_func == 'relu':
            h = tf.nn.relu(tf.matmul(x, self._w5) + self._b5)
            return h
        elif self.act_func == 'elu':
            h = tf.nn.elu(tf.matmul(x, self._w5) + self._b5)
            return h
        elif self.act_func == 'selu':
            h = tf.nn.selu(tf.matmul(x, self._w5) + self._b5)
            return h
        elif self.act_func == 'leakyrelu':
            h = tf.nn.leaky_relu(tf.matmul(x, self._w5) + self._b5)
            return h

    def set_session(self, session):
        self.session = session

    def reduced_dimension(self, x):
        h = self.encoder(x)
        return self.session.run(h, feed_dict={self._x: x})

    def reconstruct(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return self.session.run(r, feed_dict={self._x: x})

    def fit(self, x, epochs=1, batch_size=100):
        n, d = x.shape
        num_batches = n // batch_size

        obj = []
        for i in range(epochs):
            for j in range(num_batches):
                batch = x[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self.session.run([self._opt, self._loss], feed_dict={self._x: batch})
                if j % 100 == 0 and i % 100 == 0:
                    print('trainint epoch {0} batch {2} cost {1}'.format(i, ob, j))
                obj.append(ob)
        return obj


if __name__ == '__main__':
    from prepare.data_split import read_bunch, write_bunch
    import bunch
    data_set = read_bunch('e:/Paper/imb_problem/data/data_set.data')
    trX, teX, trY, teY = data_set.X_train, data_set.X_test, data_set.y_train, data_set.y_test
    x_train = trX.astype(np.float32)
    x_test = teX.astype(np.float32)
    n, m = x_train.shape
    hidden1 = 8
    hidden2 = 12
    hidden3 = 7
    hidden4 = 5
    epo = 300
    act_fun_list = ['sigmoid', 'tanh', 'elu', 'selu', 'leakyrelu']
    # act_fun_list = ['swish']
    for af in act_fun_list:
        ae = AutoEncoder(m, hidden1, hidden2, hidden3, hidden4, af)

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

        write_bunch('e:/Paper/imb_problem/data/DAF_stack_data_%s_%s.data' %
                    (af, hidden1), dataset)

