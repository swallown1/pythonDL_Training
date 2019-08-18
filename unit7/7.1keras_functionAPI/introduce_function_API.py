from keras import layers,Input,utils
from keras.models import Model
import numpy as np

#
# input_dim = Input(shape=(32,))
# dense = layers.Dense(32,activation='relu')
# output_tensor = dense(input_dim)

text_vocabulary_size = 10000  #文本的输入是一个长度可变的整数序列
question_vocabulary_size = 10000   #问题输入
answer_vocabulary_size = 500  #输出的大小

def networks_one():
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32,activation="relu")(input_tensor)
    x = layers.Dense(32,activation="relu")(x)
    output_tensor = layers.Dense(10,activation="softmax")(x)

    #Model类将输入输出张量传入变成一个模型
    model = Model(input_tensor,output_tensor)
    model.summary()

def multi_input():

    text_input = Input(shape=(None,),dtype='int32',name='text')#长度可变的整数序列
    embeded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)  # 将输入的文本嵌入到64维的向量
    ecoded_text = layers.LSTM(32)(embeded_text)  # 利用LSTM将向量编码成单个向量

    question_input = Input(shape=(None,),
                           dtype="int32",name="question")
    embeded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)  # 将输入的问题文本嵌入到64维的向量
    ecoded_question = layers.LSTM(16)(embeded_question)  # 利用LSTM将向量编码成单个向量

    concatenated = layers.concatenate([ecoded_text,ecoded_question],axis=-1)  #将编码后的文本和问题拼接起来
    answer = layers.Dense(answer_vocabulary_size,activation="softmax")(concatenated) #最后的输出用softmax进行多分类

    model = Model([text_input,question_input],answer)  #输入是模型的最开始出入 输出是整个模型的最后输出
    model.compile(optimizer='rmsprop',loss="categorical_crossentropy",metrics=['acc'])

    return model

def getData_multi_input():
    # 对对输入模型进行输入
    num_samples = 1000
    max_length = 100
    # 生成模拟的数据
    text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

    answer = np.random.randint(answer_vocabulary_size, size=(num_samples))
    # 将其转化成one-hot
    answer = utils.to_categorical(answer, answer_vocabulary_size)
    return text,question,answer

def multi_output():
    vocabulary_size = 50000
    num_income_groups = 10

    posts_input = Input(shape=(None,),dtype="int32",name="posts")
    embeded_posted = layers.Embedding(256,vocabulary_size)(posts_input)
    x = layers.Conv1D(128,5,activation="relu")(embeded_posted)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(256,5,activation="relu")(x)
    x = layers.Conv1D(256,5,activation="relu")(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(256,5,activation="relu")(x)
    x = layers.Conv1D(256,5,activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128,activation="relu")(x)

    #每个name定义了不同的输出
    age_predection = layers.Dense(1,name="age")(x)

    income_predection = layers.Dense(num_income_groups,activation="softmax",name="income")(x)

    gender_predection = layers.Dense(1,activation="softmax",name="gender")(x)
    #传入模型的输入和输出
    model = Model(posts_input,[age_predection,income_predection,gender_predection])

    # 多输出的编译选项：多重损失
    # 每个不同的输出都需要不同的训练过程，单数梯度下降是讲一个标量最小化，所以必须把多个损失合并成单个标量。
    # 方法就是对多个损失求和。
    # 在keras中，使用损失组成的列表或字典来为不同输出指定不同损失，再将损失叠加到一个全局损失上，将其最小化。
    # model.compile(optimizer='rmsprop',
    #               loss=["mse", "categorical_crossentropy", "binary_crossentropy"])
    # 或者
    # model.compile(optimizer="rmsprop",
    #               loss={'age':'mse',
    #                     'income':"categorical_crossentropy",
    #                     'gender':"binary_crossentropy"})

    # 多输出的编译选项：损失加权
    # 不平衡的损失贡献表示针对甚是最大的任务进行优先的优化，
    # 为了解决这个问题：方法是，通过对每个损失对最终损失的贡献分配不同的权重。
    # 例如对于回归问题的均方误差通常损失在3-5之间，而对于性别的分类问题的交叉熵损失可能低于0.1 所以为了
    #平衡将其分配不同的权重  让交叉熵损失权重为10 让均方误差损失权重去0.5
    model.compile(optimizer='rmsprop',
                  loss=["mse", "categorical_crossentropy", "binary_crossentropy"],
                  loss_weights=[0.25,1.,10.])
    # 或者
    # model.compile(optimizer="rmsprop",
    #               loss={'age':'mse',
    #                     'income':"categorical_crossentropy",
    #                     'gender':"binary_crossentropy"},
    #               loss_weights={'age':0.25,
    #                              'income':1.,
    #                              'gender':10.})
    return model

if __name__ == '__main__':
    networks_one()

    #问题回答模型  多输入
    text, question, answer = getData_multi_input()
    #进行训练  使用输入组合列表进行拟合
    model = multi_input()
    model.fit([text,question],answer,epochs=10,batch_size=128)
    # 或者 通过输入组成的字典来拟合
    # model.fit({'text':text,'question':question},answer,epochs=10,batch_size=128)


    #多输出模型的fit
    model2 = multi_output()
    model2.fit(posts,[age_targets,income_targets,gender_targets],epochs=10,batch_size=64)
    # model2.fit(posts,{'age':age_targets,
    #                   'income':income_targets,
    #                   'gender':gender_targets},epochs=10,batch_size=64)

