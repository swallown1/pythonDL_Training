import numpy as np

timesteps = 100
input_dim = 32
output_dim = 64

inputs = np.random.random((timesteps,input_dim))
state_t = np.zeros((output_dim)) #设置初试为0,存放细胞的上一次状态

w = np.random.random((output_dim,input_dim))
u = np.random.random((output_dim,output_dim))
b = np.random.random((output_dim))

output_res = []
for input in inputs:#没个input是  （input_dim，)张量
    out=np.tanh(np.dot(w,input)+np.dot(u,state_t)+b)
    output_res.append(out)
    state_t = out

#是（timeteps,output_dim）的二维张量
final_output_sequence = np.stack(output_res,axis=0)
print(final_output_sequence)


