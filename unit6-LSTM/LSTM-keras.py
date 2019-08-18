from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense,Embedding,LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


#数据预处理
max_features = 10000 #输入最大的维度
maxlen = 500 #序列最大长度
batch_size =32  #最多的序列数
print("加载数据......")
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

print(len(x_train),"训练序列")
print(len(x_test),"测试序列")

#转化list为array 并且取最大长度的序列
input_train = sequence.pad_sequences(x_train,maxlen=maxlen)
input_test  = sequence.pad_sequences(x_test,maxlen=maxlen)
print("x_train shape",input_train.shape)
print("x_test shape",input_test.shape)

model = Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train,y_train,
          epochs=10,
          batch_size=128,
          validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
va_loss = history.history['val_loss']

epochs =range(1,len(acc)+1)
plt.plot(epochs,acc,'ro',label="train acc")
plt.plot(epochs,val_acc,'r',label = 'validation acc')
plt.title('Train and validation acc')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'ro',label="train loss")
plt.plot(epochs,va_loss,'r',label = 'validation val_loss')
plt.title('Train and validation loss')
plt.legend()
plt.show()



