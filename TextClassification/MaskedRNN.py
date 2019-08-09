# -*- encoding: utf-8 -*-
"""

@Project: TextClassification
@File   : MaskedRNN.py
@Time   : "2019/8/5 14:51
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
import torch.nn as nn
import torch
from torch import FloatTensor


# =============================================================================
# Masked RNN
# =============================================================================
# 带mask的RNN模型
class MaskedRNN(nn.Module):
    # 1.初始化函数
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 layer_num=1,
                 drop_prob=0.0,
                 batch_first=True,
                 layer_type='LSTM',
                 bidirectional=True,
                 ):
        super(MaskedRNN, self).__init__()
        # 带mask的RNN模型
        self.info_name = 'MaskedRNN'
        self.info_task = 'None'
        self.info_func = 'Encoder or Composition'

        self.input_size = input_size  # 输入维度
        self.hidden_size = hidden_size # 隐藏层输出维度
        self.layer_num = layer_num  # 设置RNN的垂直层数
        self.drop_prob = drop_prob  # dropout 概率
        self.batch_first = batch_first  # 是否第一维为batch，根据个人习惯，默认为True
        self.seq_length = 0  # 一个batch中序列的最大长度，不同batch中该值会发生改变，初始值设为0
        self.num_direction = 2 if bidirectional else 1

        self.bidirectional = bidirectional  # 设置是否该RNN双向，默认为双向
        self.layer_type = layer_type.upper()  # 该RNN可以扩展为三种不同类型——RNN、LSTM、GRU，根据torch的定义，设置为大写字符串
        self.layer_all_types = ['RNN', 'LSTM', 'GRU']
        assert self.layer_type in self.layer_all_types  # 判定输入的RNN类型（type）是否合理
        self.rnn_cell = getattr(nn, self.layer_type+'Cell')  # 获取指定类型cell

        # multiple layers init
        # 该RNN默认为双向，由于可以设置垂直多层，每个方向需要维持层数个RNN cell
        self.forward_cells = nn.ModuleList()  # 前向cell列表
        if self.bidirectional:
            self.backward_cells = nn.ModuleList()  # 后向cell列表

        # 对cell列表中各个cell进行初始化
        # 有多少层，就有多少个cell
        for layer_id in range(self.layer_num):
            # 注意：第一层cell的输入数据维度与input相同，而后垂直方向的每一层都以上一层的输出为输入
            # 每层的输入维度为hidden_size，双向时为2*hidden_size
            # 所以第一次以后每层输入维度为hidden_size，双向时2*hidden_size
            if layer_id == 0:
                layer_input_size = self.input_size
            else:
                if self.bidirectional:
                    layer_input_size = self.hidden_size*2
                else:
                    layer_input_size = self.hidden_size
            layer_hidden_size = self.hidden_size
            self.forward_cells.append(self.rnn_cell(layer_input_size,
                                                    layer_hidden_size))
            if self.bidirectional:
                self.backward_cells.append(self.rnn_cell(layer_input_size,
                                                    layer_hidden_size))
        self.h_init = 0

    # 2.前向计算函数
    # 前向、后向操作十分相似，其实可以设计为一个函数，由传递参数控制方向
    # 但是为了表意明确，最终选择分开设计
    def _forward_compute(self,
                         cell: nn.RNNCell or nn.GRUCell or nn.LSTMCell,
                         input: torch.FloatTensor,
                         mask: torch.ByteTensor
                          ):
        # LSTM 中的cell层和hidden，详见于LSTM结构解析
        if self.layer_type == 'LSTM':
            hx, cx = self.h_init[0], self.h_init[1]
        else:
            hx = self.h_init
            cx = None
        output = []
        # LSTM 在句子维度上进行分解，序列前一状态会作为下一状态的一个输入，整个序列前部的信息不断向序列后部累积
        # mask 操作嵌入在序列计算过程中，用于摒除序列中padding值对计算的影响
        for seq_idx in range(input.size(0)):
            if self.layer_type == 'LSTM':
                now_mask = mask[seq_idx]
                hx, cx = cell(input[seq_idx], (hx, cx))
                hx = hx*now_mask+self.h_init[0]*(1-now_mask)
                cx = cx*now_mask+self.h_init[1]*(1-now_mask)
            else:
                now_mask = mask[seq_idx]
                hx = cell(input[seq_idx], hx)
                hx = hx*now_mask+self.h_init*(1-now_mask)
            output.append(hx)
        # 把序列各阶段的状态输出记录在list里，返回时使用stack函数拼接成一个tensor
        # 拼接后维度发生扩展。这里设置在第一维上扩展（seq_length，batch_size，hidden_size）
        # 考虑到后续可能作为下一次的cell的输入，设置第一维扩展较为方便（序号为0）
        all_hx = (hx, cx) if self.layer_type == 'LSTM' else hx
        output = torch.stack(output, dim=0)
        # 这里使用经典的RNN返回形式
        return output, all_hx

    # 3.后向计算函数
    # 前向、后向操作十分相似，其实可以设计为一个函数，由传递参数控制方向
    # 但是为了表意明确，最终选择分开设计
    def _backward_compute(self,
                         cell: nn.RNNCell or nn.GRUCell or nn.LSTMCell,
                         input: torch.FloatTensor,
                         mask: torch.ByteTensor
                          ):
        # LSTM 中的cell层和hidden，详见于LSTM结构解析
        if self.layer_type == 'LSTM':
            hx, cx = self.h_init[0], self.h_init[1]
        else:
            hx = self.h_init
            cx = None
        output = []
        # LSTM 在句子维度上进行分解，序列前一状态会作为下一状态的一个输入，整个序列前部的信息不断向序列后部累积
        # mask 操作嵌入在序列计算过程中，用于摒除序列中padding值对计算的影响
        for seq_idx in reversed(range(input.size(0))):
            if self.layer_type == 'LSTM':
                now_mask = mask[seq_idx]
                hx, cx = cell(input[seq_idx], (hx, cx))
                hx = hx*now_mask+self.h_init[0]*(1-now_mask)
                cx = cx*now_mask+self.h_init[1]*(1-now_mask)
            else:
                now_mask = mask[seq_idx]
                hx = cell(input[seq_idx], hx)
                hx = hx*now_mask+self.h_init*(1-now_mask)
            output.append(hx)
        # 把序列各阶段的状态输出记录在list里，返回时使用stack函数拼接成一个tensor
        # 拼接后维度发生扩展。这里设置在第一维上扩展（seq_length，batch_size，hidden_size）
        # 考虑到后续可能作为下一次的cell的输入，设置第一维扩展较为方便（序号为0）
        all_hx = (hx, cx) if self.layer_type == 'LSTM' else hx
        output = torch.stack(output, dim=0)
        # 这里使用经典的RNN返回形式
        return output, all_hx

    # 4.模型权重初始化
    def init_weight(self,
                    batch_size: int,
                    device=torch.device('cpu')):
        h0 = torch.randn(batch_size, self.hidden_size, device=device)
        self.h_init = (h0, h0) if self.layer_type == 'LSTM' else h0

    # 5.模型的前向计算定义部分
    def forward(self,
                input: torch.FloatTensor,
                mask: torch.ByteTensor,
                ):
        if self.batch_first:
            input = input.transpose(0, 1)  # seq,batch,feat
            mask = mask.transpose(0, 1)  # seq,batch
        # seq_size = input.shape[0]
        # batch_size = input.shape[1]
        # feat_size = input.shape[2]

        mask = mask.unsqueeze(2).expand(-1, -1, self.hidden_size).float()
        if not isinstance(self.h_init, FloatTensor) and not isinstance(self.h_init, tuple):
            print('Error: please invoke init weight method')
        output = None
        all_hx = []
        all_cx = []

        for layer_id in range(self.layer_num):
            # self.dropout_in()
            if self.layer_type == 'LSTM':
                forward_o, (forward_h, forward_c) = self._forward_compute(self.forward_cells[layer_id], input, mask)
                if self.bidirectional:
                    backward_o, (backward_h, backward_c) = self._backward_compute(self.backward_cells[layer_id], input, mask)
                    output = torch.cat((forward_o, backward_o), dim=-1)
                    hx = torch.cat((forward_h, backward_h), dim=-1)
                    cx = torch.cat((forward_c, backward_c), dim=-1)
                else:
                    output = forward_o
                    hx = forward_h
                    cx = forward_c
                all_hx.append(hx)
                all_cx.append(cx)
            else:
                forward_o, forward_h = self._forward_compute(self.forward_cells[layer_id], input, mask)
                if self.bidirectional:
                    backward_o, backward_h = self._backward_compute(self.backward_cells[layer_id], input, mask)
                    output = torch.cat((forward_o, backward_o), dim=-1)
                    hx = torch.cat((forward_h, backward_h), dim=-1)
                else:
                    output = forward_o
                    hx = forward_h
                all_hx.append(hx)
            input = output

        if self.layer_type == 'LSTM':
            return output, (torch.stack(all_hx,  dim=0), torch.stack(all_cx, dim=0))
        else:
            return output, torch.stack(all_hx, dim=0)

