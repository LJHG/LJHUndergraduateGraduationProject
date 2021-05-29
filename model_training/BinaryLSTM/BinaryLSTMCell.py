import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import math
import torch.nn.init as init
from utils import getLL1Mean
import time


class BinaryLSTMCell(nn.Module):
    '''
        LSTM cell
    '''
    def __init__(self, input_size, hidden_size,batch_first=True):
        super(BinaryLSTMCell, self).__init__()
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
            x = inputs[:, t, :]

            # input gate
            # calculate alpha
            # print("size of x",x.shape)
            # print("size of w_ii",self.w_ii.shape)
            # 这里的w_ii是 128*28,在求a和B时，我们把它看作是向量，1*(128*28)

            alpha_wii = getLL1Mean(self.w_ii)
            binary_wii = torch.sign(self.w_ii)

            alpha_bii = getLL1Mean(self.b_ii)
            binary_bii = torch.sign(self.b_ii)

            alpha_whi = getLL1Mean(self.w_hi)
            binary_whi = torch.sign(self.w_hi)

            alpha_bhi = getLL1Mean(self.b_hi)
            binary_bhi = torch.sign(self.b_hi)


            i = torch.sigmoid(x @ binary_wii.t()*alpha_wii + binary_bii.t()*alpha_bii
                              + h @ binary_whi.t()*alpha_whi + binary_bhi.t()*alpha_bhi)

            # forget gate

            alpha_wif = getLL1Mean(self.w_if)
            binary_wif = torch.sign(self.w_if)

            alpha_bif = getLL1Mean(self.b_if)
            binary_bif = torch.sign(self.b_if)

            alpha_whf = getLL1Mean(self.w_hf)
            binary_whf = torch.sign(self.w_hf)

            alpha_bhf = getLL1Mean(self.b_hf)
            binary_bhf = torch.sign(self.b_hf)


            f = torch.sigmoid(x @ binary_wif.t()*alpha_wif + binary_bif.t() * alpha_bif
                              + h @ binary_whf.t()*alpha_whf + binary_bhf.t()*alpha_bhf)

            # cell

            alpha_wig = getLL1Mean(self.w_ig)
            binary_wig = torch.sign(self.w_ig)

            alpha_big = getLL1Mean(self.b_ig)
            binary_big = torch.sign(self.b_ig)

            alpha_whg = getLL1Mean(self.w_hg)
            binary_whg = torch.sign(self.w_hg)

            alpha_bhg = getLL1Mean(self.b_hg)
            binary_bhg = torch.sign(self.b_hg)

            g = torch.tanh(x @ binary_wig.t()*alpha_wig + binary_big.t()*alpha_big
                           + h @ binary_whg.t()*alpha_whg + binary_bhg.t()*alpha_bhg)
            # output gate

            alpha_wio = getLL1Mean(self.w_io)
            binary_wio = torch.sign(self.w_io)

            alpha_bio = getLL1Mean(self.b_io)
            binary_bio = torch.sign(self.b_io)

            alpha_who = getLL1Mean(self.w_ho)
            binary_who = torch.sign(self.w_ho)

            alpha_bho = getLL1Mean(self.b_ho)
            binary_bho = torch.sign(self.b_ho)

            o = torch.sigmoid(x @ binary_wio.t()*alpha_wio + binary_bio.t()*alpha_bio
                              + h @ binary_who.t()*alpha_who+binary_bho.t()*alpha_bho)

            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)
            c = c_next
            h = h_next
            hidden_seq.append(h)
        hidden_seq = torch.cat(hidden_seq, dim=2).reshape(batch_size, seq_size, self.hidden_size)
        return hidden_seq, (h, c)