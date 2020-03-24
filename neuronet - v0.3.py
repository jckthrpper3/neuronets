import numpy as np
import matplotlib.pylab as plt
import random

class neuronet:
    
    def f(self, x):
        return self.sigmoid(x)
    
    def f_der(self, x):
        return self.sigmoid_der(x)
    
    def g(self, x):
        return self.softmax(x)
    
    def Loss(self, y, correct_y):
        return self.xentropy(y, correct_y)
    
    def gLoss_der(self, y, correct_y):
        return self.softmax_xenropy_der(y, correct_y)
    
    def __init__(self, layers):
        self.layers = layers
        self.size = len(layers)
        
        self.w  = [np.random.rand(layers[i+1], layers[i]) for i in range(self.size-1)]
        self.b  = [np.random.rand(layers[i+1]) for i in range(self.size-1)]
        self.h  = [None for i in range(self.size-1)]
        self.z  = [None for i in range(self.size-1)]
        
        self.dh = None
        self.l_rate = None
        
        return
    
    def predict(self, x):
        if len(x) != self.layers[0]:
            raise Exception("Invalid input format")
        
        self.z[0] = np.array(x)
        self.h[0] = self.w[0].dot(self.z[0]) + self.b[0]
        for i in range(1, self.size - 1):
            self.z[i] = self.f(self.h[i-1])
            self.h[i] = self.w[i].dot(self.z[i]) + self.b[i]
        return self.g(self.h[-1])
    
    def back_pr(self):
        for i in range(self.size-2, 0, -1):
            dh_tmp = np.transpose(self.w[i]).dot(self.dh) * self.f_der(self.h[i-1])
            
            self.db[i] += self.l_rate * self.dh
            
            self.dh = self.dh.reshape(self.layers[i+1], 1)
            self.dw[i] += self.l_rate * self.dh.dot(self.z[i].reshape(1, self.layers[i]))
            
            self.dh = dh_tmp
            
        self.db[0] += self.l_rate * self.dh
        self.dh = self.dh.reshape(self.layers[1], 1)
        self.dw[0] += self.l_rate * self.dh.dot(self.z[0].reshape(1, self.layers[0]))
        return
    
    def train_batch(self, inputs_list, targets_list, l_rate):
        self.l_rate = l_rate
        for i in range(len(inputs_list)):
            target = np.array(targets_list[i])
            
            self.dw = [np.zeros((self.layers[i+1], self.layers[i])) for i in range(self.size-1)]
            self.db = [np.zeros((self.layers[i+1],)) for i in range(self.size-1)]
            y = self.predict(inputs_list[i])
            L = self.Loss(y, target)
            self.dh = self.gLoss_der(y, target)
            self.back_pr()
            
        for i in range (self.size-1):
            self.w[i] -= self.dw[i]
            self.b[i] -= self.db[i]
        return L
    def train(self, inputs_list, targets_list, l_rate=0.3, batch_size=1):
        Loss = []
        X = []
        for i in range(len(data)//batch_size):
            j = i*batch_size
            L = self.train_batch(inputs_list[j:j+batch_size], targets_list[j:j+batch_size], l_rate)
            Loss.append(L)
            X.append(i)
        plt.plot(X, Loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()        
        return
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)
    def xentropy(self, x, y):
        return -np.sum(y*np.log(x))
    def softmax_xenropy_der(self, x, y):
        return x - y

# Do training!
A = neuronet([2, 8, 2])
data = []
train = [[[1, 1], [1, 0]], [[0.01, 0.01], [1, 0]], [[0.01, 1], [0, 1]], [[1, 0.01], [0, 1]]]

for i in range(10000):
    np.random.shuffle(train)
    for a in train:
        data.append(a)

#for i in range(10000):
#    X.append(train[random.randint(0,3)])

A.train(data, 0.5, 1)
print("Neuroxor! Insert two numbers from set {0, 1}")
L = input().split(' ')
a, b = int(L[0]), int(L[1])
L = A.predict([a, b])
p0 = L[0]*100
p1 = L[1]*100
print("%d ^ %d is\n0 with probability %.2f %%\n1 with probability %.2f %%" % (a, b, p0, p1))
