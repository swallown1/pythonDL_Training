from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding

#可以处理一批单个序列 (batch_size,samples,input_features)

model = Sequential()
model.add(Embedding(10000,32))
#返回输出序列
model.add(SimpleRNN(32,return_state=True))
model.summary()