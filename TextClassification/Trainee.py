# -*- encoding: utf-8 -*-
"""

@Project: TextClassification
@File   : Trainee.py
@Time   : "2019/8/5 8:11
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""

import torch.nn as nn
import torch
# from BiLSTMModel import Model
from RNN import Model
import torch.optim as optim


class Nebuch:
    def __init__(self, params):
        # print(params.embedding['sent'])
        self.params = params
        if self.params.has_load_model:
            self.load_model()
        else:
            self.model = Model(params)
        if params.use_gpu:
            torch.backends.cudnn.enabled=True
            self.model = self.model.to(params.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        if self.params.use_gpu:
            self.cross_entropy = self.cross_entropy.cuda()

        self.optimizer = optim.Adam(filter(lambda param: param.requires_grad,
                                         self.model.parameters()),
                                  lr=self.params.learning_rate,
                                  weight_decay=self.params.weight_decay)
        self.loss = 0.0
        self.correct_rate = 0.0
        self.best_dev_rate = 0.0
        self.best_test_rate = 0.0
        self.best_iter = 0
        self.has_save_best = self.params.has_save_model

    def clear_grads(self):
        self.model.zero_grad()

    def forward_compute(self,
                        batch_data):
        label_prob = self.model(batch_data)
        gold_label = batch_data.batch_tensor_dict["label:?"].argmax(dim=1)
        if self.model.training:
            self.loss = self.cross_entropy(label_prob, gold_label)
            self.loss.backward()
        predict_label = label_prob.argmax(dim=1)
        statistic_count = gold_label.eq(predict_label).sum()
        self.correct_rate += statistic_count.item()/batch_data.size

    def opt_step(self):
        self.optimizer.step()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def batch_show(self,
                   idx: int):
        print('current:{:^10d}, cost:{:^10f}'.format(idx + 1, self.loss.data.item()))

    def correct_show(self,
                     batch_num: int):
        print('correct rate:{:^10f}'.format(self.correct_rate/batch_num))

    def eval_rate_clear(self):
        self.correct_rate = 0

    def best_show(self,
                  iter: int):
        if iter != 0:
            print('best iter:{:^10d}, best dev:{:^10f}, test: {:^10f}'.format(self.best_iter,self.best_dev_rate,self.best_test_rate))

    def reset_best(self,
                   epoch: int,
                   dev_rate: float,
                   test_rate: float,
                   ):
        if self.best_dev_rate <= dev_rate:
            self.best_iter = epoch+1
            self.best_dev_rate = dev_rate
            self.best_test_rate = test_rate
            if self.has_save_best:
                self.save_model()

    def save_model(self):
        print(self.has_save_best)
        print('save new best model...')
        torch.save(self.model.state_dict(), self.params.save_model_path+self.model.info_name)

    def load_model(self):
        self.model.load_state_dict(torch.load((self.params.load_model_path+self.model.info_name)))
