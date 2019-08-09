# -*- coding: utf-8 -*-
"""

@Project: TextClassification
@File   : RNN.py
@Time   : "2019/8/5 8:11
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""

import torch.nn as nn
import torch
from MaskedRNN import MaskedRNN


# =============================================================================
# RNN: Recurrent Neural Network
# =============================================================================
class Model(nn.Module):
    
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.info_name = 'RNN'
        self.info_task = 'Text Classification'
        self.info_func = 'all'

        self.embed_dim = hyper_params.embed_dim
        self.has_pretrain = hyper_params.has_pretrain
        self.embed = hyper_params.embedding["sent"]
        # self.embed = nn.Embedding(self.embed.weight.shape[0], self.embed_dim)
        print(self.embed)
        if self.has_pretrain:
            self.embed_ext = hyper_params.ext_embedding["sent"]
            print(self.embed_ext)
        self.batch_size = hyper_params.batch_size

        self.hidden_size = hyper_params.hidden_size
        self.linear_size = hyper_params.hidden_size
        self.label_size = hyper_params.output_size
        self.dropout_prob = hyper_params.dropout_prob
        self.bidirectional = True

        self.lstm = MaskedRNN(self.embed_dim,
                              self.hidden_size,
                              bidirectional=self.bidirectional,
                              layer_type='LSTM',
                              layer_num=1)
        # self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.embed_dropout = nn.Dropout(self.dropout_prob)
        self.num_direction = 2 if self.bidirectional else 1
        self.linear_dropout = nn.Dropout(self.dropout_prob)
        self.linear = nn.Linear(self.linear_size*self.num_direction,
                                self.label_size,
                                bias=False)
        # self.apply(_init_weights)  # 通过递归完成所有权重的初始化
        self.init_rnn_weight()

    def init_rnn_weight(self):
        self.lstm.init_weight(self.batch_size)

    # RNN 前向计算
    def forward(self,
                batch_inst
                ):
        input = batch_inst.batch_tensor_dict["sent:A"][0]  # 1）句子输入
        mask = batch_inst.batch_tensor_dict["sent:A"][1]   # 2） mask

        # 1.Embedding 层
        if self.has_pretrain:
            # input = self.embed_ext(input)
            input = self.embed(input)+self.embed_ext(input)
        else:
            input = self.embed(input)
        if self.training:
            input = self.embed_dropout(input)

        # 2. RNN 层
        rnn_result, _ = self.lstm(input, mask)
        max_mask = mask.transpose(0, 1).unsqueeze(-1).expand_as(rnn_result)
        rnn_result = rnn_result+(max_mask.float()-1)*float("1e-4")
        out, _ = torch.max(rnn_result, dim=0)
        if self.training:
            out = self.linear_dropout(out)

        # 3. MLP层 classifier
        prob = self.linear(out)

        return prob


# 用于模型内部参数（权重，偏置等）的初始化
def _init_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0


