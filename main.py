from two_layer_net import TwoLayerNet
import numpy as np


def two_layer_net_test():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    x = np.random.rand(100, 784)
    y = net.predict(x)
    t = np.random.rand(100, 10)
    grads = net.numerical_gradient(x, t)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    two_layer_net_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
