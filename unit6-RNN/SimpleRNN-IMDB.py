from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense,Embedding,SimpleRNN
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
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['acc'])
hository = model.fit(input_train,y_train,
                     epochs=5,
                     batch_size=128,
                     validation_split=0.2)
# res = model.evaluate(input_test,y_test)
# print(res[0],res[1])

acc = hository.history['acc']
val_acc = hository.history['val_acc']
loss = hository.history['loss']
val_loss = hository.history["val_loss"]

# print(acc,val_acc)
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label="Train acc")
plt.plot(epochs,val_acc,'b',label="Train val_acc")
plt.title("train and validation acc")
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label="Train loss")
plt.plot(epochs,val_loss,'b',label="Train val_loss")
plt.title("train and validation loss")
plt.legend()
plt.show()

