import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import math
import torch.nn.init as init
import time


class LSTMCell(nn.Module):
    '''
        LSTM cell
    '''
    def __init__(self, input_size, hidden_size,batch_first=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # 输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        # cell的的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, state):
        """
        Forward
        Args:
            inputs: (batch_size, seq_size, feature_size)
            state: h0,c0 (num_layer, batch_size, hidden_size)
        """
        if (self.batch_first):
            batch_size, seq_size, _ = inputs.size()
        else:
            seq_size, batch_size, _ = inputs.size()

        if state is None:
            h = torch.zeros(1, batch_size, self.hidden_size)  # num_layer is 1
            c = torch.zeros(1, batch_size, self.hidden_size)
        else:
            (h, c) = state
            h = h
            c = c

        # squeeze过后，h,c(batch_size,hidden_size)

        hidden_seq = []

        # seq_size = 28
        for t in range(seq_size):
            start = time.time()
            x = inputs[:, t, :]

            # input gate
            i = torch.sigmoid(x @ self.w_ii.t() + self.b_ii.t() + h @ self.w_hi.t() +
                              self.b_hi.t())

            # forget gate
            f = torch.sigmoid(x @ self.w_if.t() + self.b_if.t() + h @ self.w_hf.t() +
                              self.b_hf.t())

            # cell
            g = torch.tanh(x @ self.w_ig.t() + self.b_ig.t() + h @ self.w_hg.t()
                           + self.b_hg.t())
            # output gate
            o = torch.sigmoid(x @ self.w_io.t() + self.b_io.t() + h @ self.w_ho.t() +
                              self.b_ho.t())

            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)
            c = c_next
            h = h_next
            hidden_seq.append(h)
            end = time.time()
            print(end-start)
        hidden_seq = torch.cat(hidden_seq, dim=2).reshape(batch_size, seq_size, self.hidden_size)
        return hidden_seq, (h, c)