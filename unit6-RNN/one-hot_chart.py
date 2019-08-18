import string
import numpy as np

samples = ['the cat sat on the mat.','the dog ate my homework']
characters = string.printable#可打印的ASCLL字符
#将上面所有的字符转化成字典，
# {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', ....96: '\t', 97: '\n', 98: '\r', 99: '\x0b', 100: '\x0c'}
token_index = dict(zip(range(1,len(characters)+1),characters))
max_length = 50

result = np.zeros((len(samples),max_length,max(token_index.keys())+1))
for i , sample in enumerate(samples):
    for j , character in enumerate(sample):
        index=list(token_index.keys())[list(token_index.values()).index(character)]
        result[i,j,index] = 1

print(result)