#通过API的方式可以是层组成的任意有向无环图，唯一允许的循环的是循环层内部的循环
import numpy as np
from keras import layers
from keras import Input,Model,applications

#inception 是一个卷积网络的一个架构，他的模型是堆叠的，这些模型本身看起来是小型独立网络，
# 被分为多个分支并行，最基础的形式包含3-4个分支。也有复杂的，包含池化 不同尺寸的卷积，不包含空间卷积(1*1的卷积)
#接下来介绍inception V3
def Inception_V3(x):
    if x is None:
        x = np.random.randint((4, 4, 4, 4))  # 四维向量
    branch_a = layers.Conv2D(128, 1, strides=2, activation='relu')(x)
    # 每个分支都有不步长为2 保证输出的尺寸一致，可以连接在一起
    branch_b = layers.Conv2D(128, 1, activation='relu')(x)
    branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

    branch_c = layers.AveragePooling2D(3, strides=2)(x)
    branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

    branch_d = layers.Conv2D(128, 1, activation='relu')(x)
    branch_d = layers.Conv2D(128, 3, activation='relu')(branch_c)
    branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_c)

    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d])

#残差连接
def residual_connection(x,features_size=True):
    #在keras实现残差连接的方法是恒等残差连接
    if x is None:
        x = np.random.randint((4, 4, 4, 4))  # 四维向量
    #如果特征尺寸相同
    if features_size:
        y = layers.Conv2D(128,3,activation='relu',padding='same')(x)
        y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
        y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
        y = layers.add([y,x]) #将原始特征和输出特征相加
    else:
        y = layers.Conv2D(128,3,activation='relu',padding='same')(x)
        y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
        y = layers.MaxPool2D(2,strides=2)(y)

        residual = layers.Conv2D(128,1,strides=2,padding='same')(x)
        y = layers.add([y,residual])

#深度学习中的表示瓶颈：
    # 由于在普通的网络连接，每一层都构建在前一层的激活中包含的信息。如果这一层维数太少就会出现
    # 受限于前一层信息塞入的多少，如果网络越深，那么深层次的网络就无法获取前面的相关信息，那么
    # 残差网络就可以起到这个效果。

#梯度消失
#     在LSTM中，通过引入一个细胞状态的概念，设计了一条平行于前向传播的隧道。
#     残差网络在前向传播与此类似，通过一个纯线性的携带轨道，与主要层堆叠的方向平行，跨越了
#     任意深度的层来传播。




#共享层权值，指的是可以多次重复使用一个层实例，期间在第一次使用过后，剩下的使用时都会共享之前的那个
#层的权值。
#连体LSTM 或共享LSTM 模型
def Shared_LSTM():
    lstm = layers.LSTM(32)  #实例化一个LSTM模型
    left_input = Input(shape=(None,128))
    left_out = lstm(left_input) #输入是长度为128的词向量组成的变长系列
    right_input = Input(shape=(None,128))
    right_out = lstm(right_input) #同一个LSTM实例模型 共享之前产生的权值

    #拼接之后构建一个分类器
    merged = layers.concatenate([left_out,right_out],axis=-1)
    prediction = layers.Dense(1,activation='sigmoid')(merged)

    model = Model([left_input,left_input],prediction)
    # model.fit([left_data,right_data],targets)

def Shared_Xception():
    xception_base = applications.Xception(weights= None,include_top = False)

    left_input = Input(shape=(250,250,3))
    right_input = Input(shape=(250,250,3))
    left_features = xception_base(left_input)
    right_input = xception_base(right_input)

    merged_features = layers.concatenate([left_features,right_input],axis=-1)



