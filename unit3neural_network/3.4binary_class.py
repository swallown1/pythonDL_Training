import numpy as np
from keras.datasets import imdb
from keras.layers import  Dense
from keras.models import Sequential
from keras import optimizers
import matplotlib.pyplot as plt

def get_data():
    (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
    return (train_data,train_labels),(test_data,test_labels)
#将整数序列转化成二进制矩阵
def  vectorize_sequence(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i ,sequence in enumerate(sequences):
        # print(i,sequence)
        results[i,sequence] = 1.
    return results


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = get_data()
    print(train_data[0])
    x_train = vectorize_sequence(train_data)
    x_test = vectorize_sequence(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = Sequential()
    model.add(Dense(16,input_shape=(10000,),activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    #编译模型  自定义
    model.compile(optimizer= 'rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    #验证数据 为了验证数据集
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    #使用512样本组成小批量数据，训练20次
    #监控10000个样本上的损失和精度 通过传给validation_data参数完成
    history=model.fit(partial_x_train,partial_y_train,
              epochs=20,
              batch_size=512,
              validation_data=(x_val,y_val))

    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1,len(loss)+1)
    plt.plot(epochs,loss,'ro',label="Train Loss")
    plt.plot(epochs,val_loss,'r',label="Validation Loss")
    plt.title("Train and Validation Loss")
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    # plt.show()
    #可以看出在进行第四次epochs后就出现过拟合问题，这里最好的就是在epochs=4次

    model.predict(x_test)