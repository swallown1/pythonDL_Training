import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a,b))

def naive_vector_dot(x,y):
    #判断是不是vector
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0] #两个向量必须是相同维度

    z=0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

print(naive_vector_dot(a,b))


#矩阵和向量的点积计算
def naive_matrix_vector(x,y):
    # 判断是不是maxtrix
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]  # 两个向量必须是相同维度

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] +=x[i,j]*y[j]
    return z

d = np.array([[1,2,3],[1,2,3]])
print(naive_matrix_vector(d,b))

#这里也可以用之前的向量相乘计算
# def naive_matrix_vector(x,y):
#     z = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         z[i] = naive_vector_dot(x[i,:],y)
#     return z

def naive_matrix_dot(x,y):
    # 判断是不是maxtrix
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]  # 两个向量必须是相同维度

    z = np.zeros((x.shape[0],y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            row_x = x[i,:]
            clom_y = y[:,j]
            z[i,j] = naive_vector_dot(row_x,clom_y)
    return z
e = np.array([[1,2,3],[1,2,4]])
f = np.array([[1,2,3,4],[1,2,3,5],[8,9,5,7]])
print(naive_matrix_dot(e,f))
#如果是更高维度   则要保证前者最后一个维度 和后者第一个维度一致
#(a,b,c,d) .(d,) -> (a,b,c)
#(a,b,c,d) .(d,e,f) -> (a,b,c,e,f)

















