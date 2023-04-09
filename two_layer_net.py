import sys, os
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

sys.path.append(os.pardir)


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.param['b2'] = np.zeros(hidden_size)

    def predict(self, x):
        W1, W2 = self.param['W1'], self.param['W2']
        b1, b2 = self.param['b1'], self.param['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        y = softmax(z2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
