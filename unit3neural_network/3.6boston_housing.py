from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


def getData():
    """
    404训练样本 102测试样本   每个样本13个特征
    """
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    return (train_data, train_labels), (test_data, test_labels)

def normlization(train_data,test_data):
    mean = train_data.mean(axis=0)
    train_data -=mean
    std = train_data.std(axis=0)
    train_data /=std

    test_data -=mean
    test_data /=std
    return train_data , test_data


def getModel(size):
    model = Sequential()
    #因为数据少 所以采用比较简单的模型  防止过拟合情况
    model.add(Dense(64,activation='relu',input_shape=(size,)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))
    #回归问题常采用mse做损失函数
    #训练过程需要监测一个新指标 平均绝对误差：预测值和真实值之差的绝对值
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#由于数据集较少，所以预测数据集自真的很少所以采用k折交叉验证进行训练数据集和测试数据集的划分
#k折交叉验证是将数据划分k和分区，实例化k个模型，每个模型在k-1个分区进行训练，在剩下的分区评估，模型的
#验证分数为k个模型的评估的平均值
def K_cross_validation(train_data,train_targets,k=4):
    num_val_samples = len(train_data) // k
    val_datas = []
    val_targets = []
    partial_train_datas = []
    partial_train_targets = []

    for i in range(k):
        val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
        val_target = train_targets[i*num_val_samples:(i+1)*num_val_samples]
        val_datas.append(val_data)
        val_targets.append(val_target)
        partial_train_data = np.concatenate(
            [train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
        partial_train_target = np.concatenate(
            [train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
        partial_train_datas.append(partial_train_data)
        partial_train_targets.append(partial_train_target)
    return (val_datas,val_targets),(partial_train_datas,partial_train_targets)

if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) =getData()
    train_data,test_data = normlization(train_data,test_data)
    model = getModel(train_data.shape[1])
    (val_datas, val_targets), (partial_train_datas, partial_train_targets)=K_cross_validation(train_data,train_labels)
    all_sore=[] #计算每折的验证分数
    all_mae_history = []   #存储每次mae的平均值
    num_epoch = 500
    for i in range(4):
        history = model.fit(partial_train_datas[i],partial_train_targets[i],
                            validation_data=(val_datas[i],val_targets[i]),epochs=num_epoch,batch_size=1,verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_history.append(mae_history)

        val_mse , val_mae = model.evaluate(val_datas[i],val_targets[i],verbose=0)
        all_sore.append(val_mae)
    # print(all_sore,np.mean(all_sore))
    average_mae_history = [
        np.mean([x[i]for x in all_mae_history])  for i in range(num_epoch)
    ]
    #画出每次验证的分数
    plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel("Validation MAE")
    plt.show()