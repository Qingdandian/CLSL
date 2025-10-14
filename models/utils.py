# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-9-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np


class LowRankBilinearAttention(nn.Module):
    """
    Low-rank bilinear attention network.
    """
    def __init__(self, dim1, dim2, att_dim=2048):
        """
        :param dim1: feature size of encoded images
        :param dim2: feature size of encoded labels
        :param att_dim: size of the attention network
        """
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)  # linear layer to transform encoded image
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)  # linear layer to transform decoder's output
        self.hidden_linear = nn.Linear(att_dim, att_dim)   # linear layer to calculate values to be softmax-ed
        self.target_linear = nn.Linear(att_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)  # softmax layer to calculate weights

    def forward(self, x1, x2, tau=1.0):
        """
        Forward propagation.
        :param 
            x1: a tensor of dimension (B, num_pixels, dim1)
            x2: a tensor of dimension (B, num_labels, dim2)
        """
        _x1 = self.linear1(x1).unsqueeze(dim=1)  # (B, 1, num_pixels, att_dim)
        _x2 = self.linear2(x2).unsqueeze(dim=2)  # (B, num_labels, 1, att_dim)
        t = self.hidden_linear(self.tanh(_x1 * _x2))
        t = self.target_linear(t).squeeze(-1) # B, num_labels, num_pixels
        alpha = self.softmax(t / tau) # (B, num_labels, num_pixels)
        label_repr = torch.bmm(alpha, x1)
        return label_repr, alpha
    
    
class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):  # in:num_class, out:dv
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))  # [C, dv]
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))  # [C]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):  # [Batch, Class_num, dim]
        x = input * self.weight  # [B, C, d]
        x = torch.sum(x, 2)  # [B, C]
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class JSDivergence(nn.Module):
    def __init__(self):
        super(JSDivergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p, q):
        # p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        js_div = 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
        js_div = js_div.sum(-1)
        if js_div.dim() >= 2:
            n = js_div.shape[0]
            js_div = js_div.sum() / (n * (n-1))
        else:
            js_div = js_div.mean()
        return js_div


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']  #(20, 20)
    _nums = result['nums']  #(20)
    _nums = _nums[:, np.newaxis]  #(20,1)
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    #上述是等式7
    # p = 0.2
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.diag(np.ones(num_classes) - 0.0)
    #上述是等式8
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)  # D=行和的1/2次方
    D = torch.diag(D)  # 对角化
    adj = torch.matmul(torch.matmul(A, D).t(), D)  # 对称归一化矩阵
    return adj

def gen_adj_L(A):
    sum_A_row = A.sum(1).float()
    L = torch.diag(sum_A_row) - A  # 拉普拉斯=度-A
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(L, D).t(), D)  # 对称归一化拉普拉斯矩阵
    return adj
