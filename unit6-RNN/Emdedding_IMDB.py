from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding

max_features =10000
maxlen = 20 #文本最长为20
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
# print(x_train.shape,y_train.shape)#25000  list数据

#25000 20,将列表元素转化成（sample,maxlen）的整数二维张量
x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
# print(x_train.shape)#(25000, 20)
print(x_train,y_train)
model = Sequential()
model.add(Embedding(10000,8,input_length=maxlen))
model.add(Flatten())#变成samples  maxlen*8 张量
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["acc"])
model.summary()
hository = model.fit(x_train,y_train,
          epochs=10,
          batch_size=32,
          validation_split=0.2)
res = model.evaluate(x_test,y_test)
print(res[0],res[1])
