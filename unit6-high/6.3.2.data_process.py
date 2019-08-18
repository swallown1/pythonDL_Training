# 给定5天内观测的数据  lookback=720
# 观测数据的采样是每小时一个数据点 steps = 6
# 预测未来24小时的数据  delay = 144
import climate_jena as cj
import numpy as np
#预处理是将所有数值减去均值 在除以方差
def process():
    data = cj.get_data()
    mean = data[:200000].mean(axis=0)
    # print(mean)
    data -= mean
    std = data[:200000].std(axis = 0)
    data /= std
    return data

def generator(data,lookback,delay,min_index,max_index,shuffle = False,batch_size=128,step=6):
    """"
        data 标准化过后的浮点数数据
        loohback: 输入数据包括过去多少时间步
        delay: 未来的时间不步长
        min_index,max_index：是在data中的索引，决定抽取的时间步
        shuffle: 是否打乱数据在抽取
        batch_size：每批量样本数
        steps: 数据采样的周期，设置为6 即每小时抽取一个数据点
    """
    if max_index is None:
        max_index = len(data) - delay -1
    i = min_index + lookback

    while True:
        if shuffle:
            rows = np.random.randint(min_index+lookback,max_index,batch_size)
        else:
            if i+batch_size >=max_index:
                i = min_index + lookback
            rows = np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)

        samples = np.zeros((len(rows),
                            lookback//step,
                            data.shape[-1]))
        targets = np.zeros((len(rows)))
        for j,row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j]+delay][1]

        yield samples,targets

def evaluate_naive_method(val_steps,val_data):
    batch_means=[]
    for step in range(val_steps):
        samples,targets = next(val_data)
        pred = samples[:,-1,1]
        mae = np.mean(np.abs(pred-targets))
        batch_means.append(mae)
    print(np.mean(batch_means))


if __name__ == '__main__':
    lookback =   1440
    step = 6
    delay = 144
    batch_size=128
    data =process()

    train_data = generator(data,lookback=lookback,delay=delay,min_index=0,max_index=200000,
                           shuffle=True,step=step,batch_size=batch_size)

    val_data = generator(data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,
                         step=step,batch_size=batch_size)

    test_data = generator(data, lookback=lookback,delay=delay, min_index=300001, max_index=None,
                           step=step,batch_size=batch_size)

    #为了查看验证集 需要从val_data提取一些
    val_steps = (300000-200001) //batch_size
    test_steps = (len(data)) //batch_size

    # a = np.array([[[1,2,3],[8,5,6]],[[1,2,3],[4,5,6]]])
    # print(a[:,-1,1])
    evaluate_naive_method(val_steps,val_data)