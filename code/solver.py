################################## load packages ###############################
import tensorflow as tf
import numpy as np
from model import SqueezeNet


################################## slover ###############################
class Solver(object):
    def __init__(self, conf, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.conf = conf

        ########## parameter ##########
        self.epochs = conf.epochs
        self.batch_size = conf.batch_size
        self.learning_rate = conf.learning_rate
        self.display_step = conf.display_step

        ########## data ##########
        self.classes = conf.classes
        self.height = conf.height
        self.width = conf.width
        self.channel = conf.channel

        ########## placeholder ##########
        self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.y = tf.placeholder(tf.float32, [None, self.classes])
        self.keep_prob = tf.placeholder(tf.float32)

        #### model pred 影像判断结果 ####
        self.pred = SqueezeNet(self.x, self.keep_prob, self.classes).pred

        #### loss 损失计算 ####
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        #### optimization 优化 ####
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        #### accuracy 准确率 ####
        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.pred), 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    #################### train ##################
    def train(self):
        ########## initialize variables ##########
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            step = 1

            #### epoch 世代循环 ####
            for epoch in range(self.epochs + 1):

                #### iteration ####
                for _ in range(len(self.X_train) // self.batch_size):

                    step += 1

                    ##### get x,y #####
                    batch_x, batch_y = self.random_batch(self.X_train, self.y_train, self.batch_size)

                    ##### optimizer ####
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5})

                    ##### show loss and acc #####
                    if step % self.display_step == 0:
                        loss, acc = sess.run([self.cost, self.accuracy],
                                             feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5})

                        print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))

            print("Optimizer Finished!")


    #################### test ##################
    def test(self):
        ########## initialize variables ##########
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            #### iteration ####
            for _ in range(len(self.X_test) // self.batch_size):
                ##### get x,y #####
                batch_x, batch_y = self.random_batch(self.X_test, self.y_test, self.batch_size)

                ##### show loss and acc #####
                loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0})
                print(", Minibatch Loss=" + "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            print("Test Finished!")


    ################### random generate data ###################
    def random_batch(self, images, labels, batch_size):
        '''
        :param images: 输入影像集
        :return: batch data
                 label: 影像集的label
                 输出size [N,H,W,C]
        '''

        num_images = len(images)

        ######## 随机设定待选图片的id ########
        idx = np.random.choice(num_images, size=batch_size, replace=False)

        ######## 筛选data ########
        x_batch = images[idx, :, :]

        ######## label ########
        y_batch = labels[idx, :]

        return x_batch, y_batch