from keras.preprocessing.text import Tokenizer

samples = ['the cat sat on the mat.','the dog ate my homework']

#设置分类器，提取最常见的100个词
tokenizer = Tokenizer(num_words=100)
#创建单词索引
tokenizer.fit_on_texts(samples)
#将字符串转化成整数索引列表
sequence = tokenizer.texts_to_sequences(samples)
#得到one-hot二进制
result = tokenizer.texts_to_matrix(samples,mode="binary")
word_index = tokenizer.word_index

print("找到唯一token %s" %len(word_index))



