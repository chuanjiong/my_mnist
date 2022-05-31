# -*- coding: utf-8 -*-

# mnist数据集下载地址 http://yann.lecun.com/exdb/mnist/

import numpy as np

def load_image(file_name):
    with open(file_name, 'rb') as f:
        return np.frombuffer(f.read()[0x10:], dtype=np.uint8).reshape(-1, 28, 28)

def load_label(file_name):
    with open(file_name, 'rb') as f:
        return np.frombuffer(f.read()[8:], dtype=np.uint8)

#train_data = load_image('train-images.idx3-ubyte')
#train_label = load_label('train-labels.idx1-ubyte')
test_data = load_image('t10k-images.idx3-ubyte')
test_label = load_label('t10k-labels.idx1-ubyte')

def predict_at_test_dataset(predict):
    accuracy = 0
    total = 0
    for data, label in zip(test_data, test_label):
        pred = predict(data)
        if pred == label:
            accuracy += 1
        total += 1
    print(f'accuracy: {accuracy*100/total}%, {accuracy}/{total}')

# 用猜测法来识别手写数字，准确率应该在10%

def guess1(data):
    return np.random.randint(10)

def guess2(data):
    out = np.random.randn(10)
    return np.argmax(np.exp(out)/np.sum(np.exp(out)))

predict_at_test_dataset(guess1)
predict_at_test_dataset(guess2)

