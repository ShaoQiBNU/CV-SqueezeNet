SqueezeNet详解
=============
# 一. 背景

> SqueezeNet是Han等提出的一种轻量且高效的CNN模型，在ImageNet上实现了AlexNet级精度，参数却减少了50倍，是AlexNet的1/50。此外，通过模型压缩技术，能够将SqueezeNet压缩到小于0.5MB（比AlexNet小510倍）。
>
> 具有相同精度的CNN模型，较小的CNN架构至少有以下三个优点：
>
> (1) 更高效的分布式训练，小模型参数小，网络通信量减少；
>
> (2) 便于模型更新，模型小，客户端程序容易更新，减少从云端下载模型的带宽；
>
> (3) 较小的CNN更适合部署在FPGA和内存有限的硬件上；

# 二. 网络结构

## (一) 设计理念

> Han等将CNN模型设计的研究总结为四个方面：
>
> (1) **模型压缩**：对pre-trained的模型进行压缩，使其变成小模型，如采用网络剪枝和量化等手段；
>
> (2) **CNN微观结构**：对单个卷积层进行优化设计，如采用1x1的小卷积核，还有很多采用可分解卷积（factorized convolution）结构或者模块化的结构（blocks，modules）；
>
> (3) **CNN宏观结构**：网络架构层面上的优化设计，如网路深度（层数），还有像ResNet那样采用“短路”连接（bypass connection）；
>
> (4) **设计空间**：不同超参数、网络结构，优化器等的组合优化。

> SqueezeNet也是从这四个方面来进行设计的，其设计理念可以总结为以下三点：
>
> (1) 用1x1卷积核替换3x3卷积核，通道数相同的情况下，1x1的卷积核参数要比3x3的卷积核减少9倍。
>
> (2) 减少3x3卷积核的输入通道数（input channels），因为卷积核参数为：(number
> of input channels) * (number of filters) * 3 * 3，使用瓶颈层减少通道数的话参数就自然少了很多。 
>
> (3) 延迟下采样（downsample），这样前面的layers可以有较大的激活的特征图，其保留了更多的信息，有利于提升模型准确度。目前下采样一般采用strides>1的卷积层或者pool layer。

## (二) 结构

### 1. Fire模块

> SqueezeNet网络基本单元是采用了模块化的卷积，其称为Fire module。Fire module主要包含两层卷积操作：
>
> (1) 采用 1x1 卷积核的squeeze层；
>
> (2) 混合使用 1x1 和 3x3 卷积核的expand层；
>
> Fire模块的基本结构如图所示。在squeeze层卷积核数记为 <a href="https://www.codecogs.com/eqnedit.php?latex=s_{1x1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{1x1}" title="s_{1x1}" /></a>，在expand层，记 1x1 卷积核数为 <a href="https://www.codecogs.com/eqnedit.php?latex=e_{1x1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?e_{1x1}" title="e_{1x1}" /></a> ，而3x3卷积核数为 <a href="https://www.codecogs.com/eqnedit.php?latex=e_{3x3}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?e_{3x3}" title="e_{3x3}" /></a>。这三个参数为超参数，其中设定：<a href="https://www.codecogs.com/eqnedit.php?latex=s_{1x1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{1x1}" title="s_{1x1}" /></a>的值小于 <a href="https://www.codecogs.com/eqnedit.php?latex=e_{1x1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?e_{1x1}" title="e_{1x1}" /></a>与 <a href="https://www.codecogs.com/eqnedit.php?latex=e_{3x3}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?e_{3x3}" title="e_{3x3}" /></a>的和，这样有助于限制 3x3 过滤器的输入通道数量，也就是expand层的输入特征图的通道数。

![image](https://github.com/ShaoQiBNU/CV-SqueezeNet/blob/master/images/1.png)

### 2. 整体设计

> SqueezeNet网络结构如图所示：

![image](https://github.com/ShaoQiBNU/CV-SqueezeNet/blob/master/images/2.png)

> 左图是标准的SqueezeNet， 从一个独立的卷积层(conv1) 开始，然后是8个Fire模块 (fire2-9)， 最后一个卷积层(conv10)。 从网络的开始到结束，逐渐增加每个Fire模块的过滤器数量 。其中穿插着 stride=2 的 maxpool层，其主要作用是下采样，并且采用延迟的策略，尽量使前面层拥有较大的 feature map。中图和右图使用了 ResNet 网络中的 shortcut 作为提升策略。各层具体参数设计如图所示：

![image](https://github.com/ShaoQiBNU/CV-SqueezeNet/blob/master/images/3.png)

> SqueezeNet的详细信息和设计选择如下：
>
> (1) 在Fire模块中，expand层采用了混合卷积核1x1和3x3，其stride均为1，对于1x1卷积核，其输出feature map与原始一样大小，但是由于它要和3x3得到的feature map做concat，所以3x3卷积进行了padding=1的操作，实现的话就设置padding="same"；
>
> (2) Fire模块中squeeze层和expand层的激活函数采用ReLU；
>
> (3) Fire9层后采用了dropout，其中keep_prob=0.5；
>
> (4) 没有全连接层，而是采用了全局的avgpool层，即pool size与输入feature map大小一致；
>
> (5) 训练采用线性递减的学习速率，初始学习速率为0.04，整个训练中线性降低学习率；

### 3. 超参数

> 在SqueezeNet中，每一个Fire module有3个维度的超参数，即<a href="https://www.codecogs.com/eqnedit.php?latex=s_{1x1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?s_{1x1}" title="s_{1x1}" /></a> 、 <a href="https://www.codecogs.com/eqnedit.php?latex=e_{1x1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?e_{1x1}" title="e_{1x1}" /></a> 和 <a href="https://www.codecogs.com/eqnedit.php?latex=e_{3x3}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?e_{3x3}" title="e_{3x3}" /></a>。SqueezeNet一共有8个Fire modules，即24个超参数。下面两个是需要注意的比例关系： 
>
> (1) SR：压缩比，即the squeeze ratio ，为squeeze层中filter个数除以Fire module中filter总个数得到的一个比例。 
>
> (2) pct3x3：在expand层有1x1和3x3两种卷积，这里定义的参数是3x3卷积个数占卷积总个数的比例。 
>
> 分别测试SR与模型准确率以及模型大小的关系、pct3x3与模型准确率以及模型大小的关系。如下图可知，左图给出了压缩比（SR）的影响。压缩比小于0.25时，正确率开始显著下降。右图给出了3x3卷积比例的影响，在比例小于25%时，正确率开始显著下降，此时模型大小约为原先的44%。超过50%后，模型大小显著增加，但是正确率不再上升。

![image](https://github.com/ShaoQiBNU/CV-SqueezeNet/blob/master/images/4.png)

# 三. 代码

> 利用标准SqueezeNet网络结构实现CIFAR 10分类的数据集分类，代码在[code][https://github.com/ShaoQiBNU/CV-SqueezeNet/tree/master/code]，具体如下：

## 1. main

```python
################################## load packages ###############################
import tensorflow as tf
import numpy as np
import argparse
import cifar10
from solver import Solver


###################### load data #########################
def load_data():

    ################ download dataset ####################
    cifar10.maybe_download_and_extract()

    ################ load train and test data ####################
    images_train, _, labels_train = cifar10.load_training_data()
    images_test, _, labels_test = cifar10.load_test_data()

    return images_train, labels_train, images_test, labels_test


################################## main ###############################
if __name__ == "__main__":

    ###################### load train and test data #########################
    images_train, labels_train, images_test, labels_test = load_data()

    ###################### argument ####################
    parser = argparse.ArgumentParser()

    ############# parameter ##############
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--display_step', type=int, default=20)

    ############# data #############
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--channel', type=int, default=3)


    ############# conf #############
    conf = parser.parse_args()

    ###################### solver ####################
    solver = Solver(conf, images_train, labels_train, images_test, labels_test)

    ############# train #############
    solver.train()

    ############# test #############
    solver.test()
```

## 2. model

```python
################################## load packages ###############################
import tensorflow as tf
from layer import *


################################## SqueezeNet ###############################
class SqueezeNet(object):
    def __init__(self, x, keep_prob, classes):

        ########## conv1 ##########
        net = tf.layers.conv2d(x, 96, [7, 7], strides=[2, 2], padding="SAME",
                               activation=tf.nn.relu, name="conv1")

        ########## maxpool1 ##########
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2], name="maxpool1")

        ########## fire2 ##########
        net = self._fire(net, 16, 64, "fire2")

        ########## fire3 ##########
        net = self._fire(net, 16, 64, "fire3")

        ########## fire4 ##########
        net = self._fire(net, 32, 128, "fire4")

        ########## maxpool4 ##########
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2], name="maxpool4")

        ########## fire5 ##########
        net = self._fire(net, 32, 128, "fire5")

        ########## fire6 ##########
        net = self._fire(net, 48, 192, "fire6")

        ########## fire7 ##########
        net = self._fire(net, 48, 192, "fire7")

        ########## fire8 ##########
        net = self._fire(net, 64, 256, "fire8")

        ########## maxpool8 ##########
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2], name="maxpool8")

        ########## fire9 ##########
        net = self._fire(net, 64, 256, "fire9")

        ########## dropout ##########
        net = tf.layers.dropout(net, keep_prob)

        ########## conv10 ##########
        net = tf.layers.conv2d(net, classes, [1, 1], strides=[1, 1], padding="SAME",
                               activation=tf.nn.relu, name="conv10")

        ########## avgpool10 ##########
        # net = tf.layers.average_pooling2d(net, [13, 13], strides=[1, 1], name="avgpool10")
        net = tf.layers.average_pooling2d(net, [1, 1], strides=[1, 1], name="avgpool10")


        ########## squeeze the axis ##########
        net = tf.squeeze(net, axis=[1, 2])

        self.logits = net
        self.pred = tf.nn.softmax(net)


    ###################### fire module #########################
    def _fire(self, inputs, squeeze_depth, expand_depth, scope):

        with tf.variable_scope(scope):

            ########## squeeze ##########
            squeeze = tf.layers.conv2d(inputs, squeeze_depth, [1, 1],
                                       strides=[1, 1], padding="SAME",
                                       activation=tf.nn.relu, name="squeeze")

            ################ expand ################
            ########## expand 1x1 ##########
            expand_1x1 = tf.layers.conv2d(squeeze, expand_depth, [1, 1],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_1x1")

            ########## expand 3x3 ##########
            expand_3x3 = tf.layers.conv2d(squeeze, expand_depth, [3, 3],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_3x3")

            ########## concat ##########
            return tf.concat([expand_1x1, expand_3x3], axis=3)
```

## 3. solver

```python
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
```
