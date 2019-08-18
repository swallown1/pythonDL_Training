import numpy as np

samples = ['the cat sat on the mat.','the dog ate my homework']
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            #没有给编号0指定单词
            token_index[word] = len(token_index)+1

max_length = 10
#之前0没有指定单词，所以这里要+1
result = np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))
for i , sample in enumerate(samples):
    for j , word in list(enumerate(sample.split()))[:6]:
        index = token_index.get(word)
        result[i,j,index] = 1

print(result)