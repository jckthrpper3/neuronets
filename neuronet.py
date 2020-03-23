import numpy as np

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
            #print(np.transpose(self.w[i]).shape, self.dh.shape)
            dh_tmp = np.transpose(self.w[i]).dot(self.dh) * self.f_der(self.h[i-1])
            
            self.db[i] += self.l_rate * self.dh
            
            self.dh = self.dh.reshape(self.layers[i+1], 1)
            self.dw[i] += self.l_rate * self.dh.dot(self.z[i].reshape(1, self.layers[i]))
            
            self.dh = dh_tmp
            
        self.db[0] += self.l_rate * self.dh
        self.dh = self.dh.reshape(self.layers[1], 1)
        self.dw[0] += self.l_rate * self.dh.dot(self.z[0].reshape(1, self.layers[0]))
        return
    
    def learn(self, x, correct_y, l_rate):
        self.l_rate = l_rate
        correct_y = np.array(correct_y)
        
        self.dw = [np.zeros((self.layers[i+1], self.layers[i])) for i in range(self.size-1)]
        self.db = [np.zeros((self.layers[i+1],)) for i in range(self.size-1)]
        y = self.predict(x)
        L = self.Loss(y, correct_y)
        self.dh = self.gLoss_der(y, correct_y)
        self.back_pr()
        for i in range (self.size-1):
            self.w[i] -= self.dw[i]
            self.b[i] -= self.db[i]
        return L
    
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
train = [[[1, 1], [1, 0]], [[0.01, 0.01], [1, 0]], [[0.01, 1], [0, 1]], [[1, 0.01], [0, 1]]]
for i in range(1000):
    np.random.shuffle(train)
    for data in train:
        A.learn(*data, 0.2)

print("Neuroxor! Insert two numbers from set {0, 1}")
L = input().split(' ')
a, b = int(L[0]), int(L[1])
L = A.predict([a, b])
p0 = L[0]*100
p1 = L[1]*100
print("%d ^ %d is\n0 with probability %.2f %%\n1 with probability %.2f %%" % (a, b, p0, p1))
