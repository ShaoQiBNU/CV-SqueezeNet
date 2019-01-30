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


