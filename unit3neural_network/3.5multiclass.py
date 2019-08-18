from keras.datasets import reuters
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def getData():
    #8982训练样本 2246测试样本   每个样本是整数列表(表示单词索引)  标签是0-45 对应个话题的索引
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    return (train_data, train_labels), (test_data, test_labels)

def transpose_word(train_data):
    word_index = reuters.get_word_index()
    revese_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = ' '.join([revese_word_index.get(i - 3, '?') for i in train_data[0]])
    return decoded_newswire

def to_onehot(labels,dim=46):
    results = np.zeros((len(labels),dim))
    for  i, label in enumerate(labels):
        results[i,label] = 1
    return  results

# 进行数据处理，将整数型向量转化成二进制向量
def vectorize_sequences(sequences,dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence] = 1
    return  results

def plot(history):
    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'ro', label="Train Loss")
    plt.plot(epochs, val_loss, 'r', label="Validation Loss")
    plt.title("Train and Validation Loss")
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = getData()
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    #y_train = to_categorical(train_labels)
    #类似于
    y_train = to_onehot(train_labels)
    y_test = to_onehot(test_labels)
    # print(y_train[0])

    #定义模型
    # model = Sequential()
    # #因为在这个问题中 其分类的种类有64种，所以来说，
    # # 中间层如果小于这个容易造成信息瓶颈，所以设定中间层有64个神经元
    # #最后用softmax为了得到每个分类的概率分布
    # model.add(Dense(64,activation="relu",input_shape=(10000,)))
    # model.add(Dense(64,activation="relu"))
    # model.add(Dense(64,activation="relu"))
    # model.add(Dense(46,activation="softmax"))
    #
    # #对于多分类问题最好的优化器是分类交叉熵函数
    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',metrics=['acc'])
    # #验证数据集
    # x_val = x_train[:1000]
    # partial_x_train = x_train[1000:]
    #
    # y_val = y_train[:1000]
    # partial_y_train = y_train[1000:]
    # # print(x_train.shape,partial_y_train.shape)
    # #开始训练  循环20次  以512个样本组成一个小批量
    # history = model.fit(partial_x_train, partial_y_train,
    #                     epochs=20,
    #                     batch_size=512,
    #                     validation_data=(x_val, y_val))
    #
    # # plot(history)
    # res= model.evaluate(x_test,y_test)
    # print(res)