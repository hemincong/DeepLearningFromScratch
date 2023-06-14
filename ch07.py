from common.util import im2col

class Convolutions:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH)) / self.stride
        out_w = int(1 + (W + 2 * self.pad - FW)) / self.stride

        col = im2col(X)
