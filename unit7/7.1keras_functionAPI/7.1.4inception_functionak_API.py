#通过API的方式可以是层组成的任意有向无环图，唯一允许的循环的是循环层内部的循环
import numpy as np
from keras import layers

#inception 是一个卷积网络的一个架构，他的模型是堆叠的，这些模型本身看起来是小型独立网络，
# 被分为多个分支并行，最基础的形式包含3-4个分支。也有复杂的，包含池化 不同尺寸的卷积，不包含空间卷积(1*1的卷积)
#接下来介绍inception V3
# def Inception_V3():
#