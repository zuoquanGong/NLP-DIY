# -*- encoding: utf-8 -*-
"""

@Project: TextClassification
@File   : Nebuchadnezzar_master.py
@Time   : "2019/8/5 21:16
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
from SimpleFeeder.SimpleFeeder import InstanceList
from Trainee import Nebuch


class Nebuchadnezzar:
    def __init__(self,
                 params):
        self.params = params
        self.train = InstanceList(is_train=True)
        self.dev = InstanceList()
        self.test = InstanceList()
        self.trainee = None

    # 1.数据预处理
    def data_process(self):
        # print('separator:', self.params.separator)
        self.train.read(self.params.train_path, self.params.read_model, self.params.reader, separator=self.params.separator,
                        no_fix=self.params.no_fix, char_filter=self.params.char_filter, line_filter_mark=self.params.line_filter_mark,
                        txt_filter=self.params.txt_filter, has_head=self.params.has_head)
        self.dev.read(self.params.dev_path, self.params.read_model, self.params.reader, separator=self.params.separator,
                        no_fix=self.params.no_fix, char_filter=self.params.char_filter, line_filter_mark=self.params.line_filter_mark,
                      txt_filter=self.params.txt_filter, has_head=self.params.has_head)
        self.test.read(self.params.test_path, self.params.read_model, self.params.reader, separator=self.params.separator,
                       no_fix=self.params.no_fix, char_filter=self.params.char_filter, line_filter_mark=self.params.line_filter_mark,
                       txt_filter=self.params.txt_filter, has_head=self.params.has_head)

        self.train.build_vocab()
        if self.params.has_pretrain:
            self.train.load_sent_pretrain(self.params.pretrain_path)
            # 这里我们维持了两套 word-embedding ，输入模型的是外部 embedding 和语料 embedding 的加和
            # 外部引入的不会随着模型训练而改变，语料embedding会随训练而改变

        print('train:')
        print(self.train[0])
        print(self.train[100])
        print('dev:')
        print(self.dev[0])
        print(self.dev[100])
        print('test:')
        print(self.test[0])
        print(self.test[100])

        self.train.txt2idx()

        print('train:')
        print(self.train[0])
        print(self.train[100])
        print('dev:')
        print(self.dev[0])
        print(self.dev[100])
        print('test:')
        print(self.test[0])
        print(self.test[100])
        # for w in self.test[0].attr['sent:A'].value:
        #     print(self.test[0].attr['sent:A'].vocab_group['sent'].id2item[w])
        # print(self.test[0].attr['label:?'].vocab_group['label'].id2item[self.test[0].attr['label:?'].value[0]])

        self.train.build_sent_pretrain(self.params.embed_dim)
        print('train data set size: ', len(self.train))
        print('dev data set size: ', len(self.dev))
        print('test data set size: ', len(self.test))
        self.train.show_labels()

    # 2.模型训练
    def train_process(self):
        # 以下两行为 ESIM 模型传递 embeding
        # self.params.embedding = self.train.embedding
        # if self.params.has_pretrain:
        #     self.params.ext_embedding = self.train.ext_embedding
        # else:
        #     self.params.ext_embedding = None
        self.params.embedding = self.train.get_pretrain()
        self.params.ext_embedding = self.train.get_pretrain_ext()
        if self.params.show_interval == 0:
            self.params.show_interval = len(self.train)//self.params.batch_size//self.params.show_times
        # print(self.params.show_interval)
        self.params.output_size = self.train.label_size()

        self.trainee = Nebuch(self.params)
        for epoch in range(self.params.epochs):
            print('\n[  Iteration %s  ]\n' % str(epoch+1))
            self.train.shuffle()
            self.trainee.train()
            self.trainee.eval_rate_clear()
            for update_iter, batch_data in enumerate(
                    self.train.batch_generator(
                        self.params.batch_size,
                        self.params.use_gpu,
                        device=self.params.device
                    )
            ):

                self.trainee.clear_grads()
                self.trainee.forward_compute(batch_data)
                self.trainee.opt_step()

                if update_iter % self.params.show_interval == 0 and update_iter != 0:
                    self.trainee.batch_show(update_iter)

            self.trainee.correct_show(self.train.batch_num)
            self.trainee.eval()
            print('dev data set: ')
            dev_rate = self.eval_process(self.dev)
            print('test data set: ')
            test_rate = self.eval_process(self.test)
            self.trainee.reset_best(epoch, dev_rate, test_rate)
            self.trainee.best_show(epoch)

    # 3.模型测试
    def eval_process(self,
                     dataset: InstanceList
                     ) -> float:
        self.trainee.eval_rate_clear()
        for update_iter, batch_data in enumerate(
                dataset.batch_generator(self.params.batch_size, self.params.use_gpu,device=self.params.device)):
            self.trainee.forward_compute(batch_data)
        self.trainee.correct_show(dataset.batch_num)
        return self.trainee.correct_rate/dataset.batch_num

