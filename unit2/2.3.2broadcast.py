#如果两个张量的形状不同 小的张量会被广播
import numpy as np

def naive_add_matrix_and_vectoer(x,y):
    assert len(x.shape) == 2 #x是一个2D的张量
    assert len(y.shape) == 1 #y是一个向量
    assert x.shape[1] == y.shape[0] #维度保持一致

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[j]

    return x

print(naive_add_matrix_and_vectoer(np.array([[1,2,3],[1,2,3]]),np.array([5,5,5])))