import numpy as np

#标量（0D张量）包含一个数字的张量
x = np.array(2)
print(x.ndim)  #ndim 维度称为轴

#向量（1D张量）包含一个数组的张量
x = np.array([1,2,3])
print(x.ndim)

#矩阵（2D张量）向量组成的数组
x = np.array([[1,2,3,4],
              [4,5,6,7]])
print(x.ndim)  #ndim 维度

#
# 关键属性
# 1.轴的个数(阶)：ndim   3D张量有3个轴 矩阵有2个轴
# 2.形状:shape   (3,3,5)  (2,4) (5,) ( )
# 3.数据类型：dtype

# a = np.array([[1,2,3,4,5,6,7,8,9],
#              [5,5,6,9,8,4,2,3,1]])
# print(a[2])


# 现实中的张量
#     2D (samples,feature)
#     时间序列说或序列数据   3D (samples,timesteps,feature)
#     图像：4D  (samples,height,width,channels) 或 (samples,channels,height,width)
#     视频：5D  (samples,frames,height,width,channels) 或 (samples,frames,channels,height,width)

def naive_add(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] +=y[i,j]

    return x

print(naive_add(np.array([[1,2],[1,2]]),np.array([[3,4],[3,4]])))



