import keras
import numpy as np
import random ,sys

maxlen = 60
step = 3
sentances = []
next_chars = []

def chart_to_vector():

    #'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
    path = keras.utils.get_file('nietzsche.txt',origin='./data/')
    text = open(path).read().lower()

    for i in range(0,len(text)-maxlen,step):
        sentances.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])
    print("Numbers of sequcenes is ",len(sentances))

    chars = sorted(list(set(text)))
    char_index = dict((char, chars.index(char)) for char in chars)

    x = np.zeros((len(sentances),maxlen,len(chars)),dtype=np.bool)
    y = np.zeros((len(sentances),len(chars)),dtype=np.bool)
    for i , sentance in enumerate(sentances):
        for j , char in enumerate(sentance):
            x[i,j,char_index[char]] = 1
        y[i,char_index[next_chars[i]]] = 1
    return x,y,chars,char_index

def get_model(chars):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(128,input_shape=(maxlen,len(chars))))
    model.add(keras.layers.Dense(len(chars),activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
    return model

def sample(preds , temp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds/ np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

if __name__ == '__main__':
    path = keras.utils.get_file('nietzsche.txt', origin='./data/')
    text = open(path).read().lower()
    x, y, chars, char_index = chart_to_vector()

    #获取model
    model = get_model(chars=chars)

    #训练 循环生成样本
    for i in range(1,60):
        print('epoch',i)
        model.fit(x,y,batch_size=128,epochs=1)
        #产生一个文本种子
        start_index = random.randint(0,len(text)-maxlen-1)
        generated_text = text[start_index:start_index+maxlen]
        print('——Generating with seed: "'+generated_text+'"')

    for temperature in [0.2,0.5,1.0,1.2]:#尝试不同的温度
        print("---- temperature",temperature)
        sys.stdout.write(generated_text)
        #通过不同的温度，生成400个字符
        for i in range(400):
            sampled = np.zeros((1,maxlen,len(chars)))
            #将生成的字符进行one-hot 编码
            for t , char in enumerate(generated_text):
                sampled[0,t,char_index[char]] = 1

            #通过model预测下一个字符
            preds = model.predict(sampled,verbose=0)[0]
            next_index = sample(preds,temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
        print('\n')