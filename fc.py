# -*- coding: utf-8 -*-

# mnist数据集下载地址 http://yann.lecun.com/exdb/mnist/

import random
import numpy as np

def load_image(file_name):
    with open(file_name, 'rb') as f:
        return np.frombuffer(f.read()[0x10:], dtype=np.uint8).reshape(-1, 784)

def load_label(file_name):
    with open(file_name, 'rb') as f:
        return np.frombuffer(f.read()[8:], dtype=np.uint8)

def print_image(image):
    x = image.reshape(28,28)
    for j in range(28):
        for i in range(28):
            if x[j][i] > 127:
                print('*', end='')
            else:
                print(' ', end='')
        print('\n', end='')

train_data = load_image('train-images.idx3-ubyte')
train_label = load_label('train-labels.idx1-ubyte')
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

# 用三层全连接网络来识别手写数字

def prepare(x):
    return x/255-0.5

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1-sigmoid(x))

class neural_network_2layers():
    def __init__(self):
        self.input_layer = 784
        self.output_layer = 10
        self.learning_rate = 0.01
        self.w1 = np.random.randn(self.output_layer, self.input_layer)
        self.b1 = np.random.randn(self.output_layer)
        
    def forward(self, x):
        self.input = prepare(x)
        self.z1 = self.w1 @ self.input + self.b1
        self.output = sigmoid(self.z1)
        return self.output
        
    def backward(self, out, y):
        d_output = out - np.eye(10)[y]
        d_z1 = d_output * sigmoid_derivate(self.z1)
        d_w1 = d_z1.reshape(-1,1) * self.input
        d_b1 = d_z1
        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1
        
    def predict(self, x):
        return np.argmax(self.forward(x))
        
    def validate(self, s):
        accuracy = 0
        for x, y in s:
            y_pred = self.predict(x)
            if y_pred == y:
                accuracy += 1
        return accuracy, len(s)
        
    def train(self, epochs=50, valid_set_size=2000):
        train_set = [(data, label) for data, label in zip(train_data[valid_set_size:], train_label[valid_set_size:])]
        valid_set = [(data, label) for data, label in zip(train_data[:valid_set_size], train_label[:valid_set_size])]
        for epoch in range(epochs):
            random.shuffle(train_set)
            for x, y in train_set:
                out = self.forward(x)
                self.backward(out, y)
            accuracy, total = self.validate(valid_set)
            print(f'2 layers epoch: {epoch} accuracy: {accuracy*100/total}%')

class neural_network_3layers():
    def __init__(self):
        self.input_layer = 784
        self.hidden_layer = 128
        self.output_layer = 10
        self.learning_rate = 0.01
        self.w1 = np.random.randn(self.hidden_layer, self.input_layer)
        self.b1 = np.random.randn(self.hidden_layer)
        self.w2 = np.random.randn(self.output_layer, self.hidden_layer)
        self.b2 = np.random.randn(self.output_layer)
        
    def forward(self, x):
        self.input = prepare(x)
        self.z1 = self.w1 @ self.input + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.w2 @ self.a1 + self.b2
        self.output = sigmoid(self.z2)
        return self.output
        
    def backward(self, out, y):
        d_output = out - np.eye(10)[y]
        d_z2 = d_output * sigmoid_derivate(self.z2)
        d_w2 = d_z2.reshape(-1,1) * self.a1
        d_b2 = d_z2
        d_a1 = self.w2.T @ d_z2
        d_z1 = d_a1 * sigmoid_derivate(self.z1)
        d_w1 = d_z1.reshape(-1,1) * self.input
        d_b1 = d_z1
        self.w2 -= self.learning_rate * d_w2
        self.b2 -= self.learning_rate * d_b2
        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1
        
    def predict(self, x):
        return np.argmax(self.forward(x))
        
    def validate(self, s):
        accuracy = 0
        for x, y in s:
            y_pred = self.predict(x)
            if y_pred == y:
                accuracy += 1
        return accuracy, len(s)
        
    def train(self, epochs=50, valid_set_size=2000):
        train_set = [(data, label) for data, label in zip(train_data[valid_set_size:], train_label[valid_set_size:])]
        valid_set = [(data, label) for data, label in zip(train_data[:valid_set_size], train_label[:valid_set_size])]
        for epoch in range(epochs):
            random.shuffle(train_set)
            for x, y in train_set:
                out = self.forward(x)
                self.backward(out, y)
            accuracy, total = self.validate(valid_set)
            print(f'3 layers epoch: {epoch} accuracy: {accuracy*100/total}%')


nn2 = neural_network_2layers()

# 训练前
print('2 layers before train')
predict_at_test_dataset(nn2.predict)

nn2.train()

# 训练后
print('2 layers after train')
predict_at_test_dataset(nn2.predict)


nn3 = neural_network_3layers()

# 训练前
print('3 layers before train')
predict_at_test_dataset(nn3.predict)

nn3.train()

# 训练后
print('3 layers after train')
predict_at_test_dataset(nn3.predict)

