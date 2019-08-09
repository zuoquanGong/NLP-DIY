# -*- encoding: utf-8 -*-
"""

@Project: TextClassification
@File   : config.py
@Time   : "2019/8/5 8:52
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
from SimpleFeeder.Readers import SICK_reader
from optparse import OptionParser


class Params:
    def __init__(self):
        # self.use_gpu = True
        self.use_gpu = False
        self.device_id = 0
        self.device = 'cpu'

        # SimpleFeeder 参数：
        self.read_model = ['label:?', 'sent:A']
        # self.read_model = ['label:entail', 'sent:A', 'sent:B']+11*['']
        self.reader = SICK_reader
        self.no_fix = False  # 如果是与SICK集相同：第一行有表头，以‘\t’为分隔符separator，空格为minicut，则可以选择no_fix
        # self.char_filter = ['(', ')']
        self.has_head = False
        self.char_filter = []
        self.separator = ' ||| '
        self.minicut = ' '
        self.line_filter_mark = '-'  # 句子开头为此符号的将被剔除
        self.txt_filter = txt_filter

        self.path = 'data/SST-1/'
        self.train_path = self.path+'train.txt'
        # self.train_path = 'data/snli_1.0_test.txt'
        self.dev_path = self.path+'dev.txt'
        # self.dev_path = 'data/snli_1.0_dev.txt'
        self.test_path = self.path+'test.txt'
        # self.test_path = 'data/snli_1.0_test.txt'

        # self.vocab_freq_cut = 0  # 根据词频剪出词表
        # self.max_sent_len = 80

        self.has_pretrain = True
        # self.has_pretrain = False
        self.pretrain_path = r'G:\PycharmWorkStation\Nebuchadnezzar\pretrain\word2vec_40w_300.txt'
        # 注：预加载的词向量数不大于20000，可以在SimpleFeeder.InstanceList.load_pretrain 中修改

        self.dropout_prob = 0.3
        self.hidden_size = 300
        self.output_size = 0
        self.embed_dim = 300

        self.epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-7
        self.batch_size = 32
        self.show_interval = 0
        self.show_times = 5  # 当 show_interval 不为 0 时，该参数无效

        self.has_load_model = False
        self.has_save_model = False
        self.save_model_path = 'model_save/'
        self.load_model_path = 'model_save/'

    def arg_parse(self):
        parser = OptionParser()
        parser.add_option("--train", dest="train_path",
                          default="", help="train dataset")
        parser.add_option("--dev", dest="dev_path",
                          default="", help="dev dataset")
        parser.add_option("--test", dest="test_path",
                          default="", help="test dataset")
        parser.add_option("--lr", dest="lr", type=float,
                          default=0.0, help="learning rate")
        parser.add_option("--wd", dest="weight_decay", type=float,
                          default=0.0, help="weight decay")
        parser.add_option("--hiddenSize", dest="hidden_size", type=int,
                          default=0, help="hidden size")
        parser.add_option("--dropout", dest="dropout_prob", type=float,
                          default=0.0, help="dropout prob")
        parser.add_option("--batchSize", dest="batch_size", type=int,
                          default=0, help="batch size")
        parser.add_option("--epoch", dest="epoch", type=int,
                          default=0, help="epoch")
        parser.add_option("--device", dest="device", type=str,
                          default="cpu", help="device[‘cpu’,‘cuda:0’,‘cuda:1’,......]")
        parser.add_option("--showIter", dest="show_interval", type=int,
                          default=0, help="show interval")
        parser.add_option("--showTimes", dest="show_times", type=int,
                          default=0, help="show times")
        (options, args) = parser.parse_args()
        if options.train_path:
            self.train_path = options.train_path
        if options.dev_path:
            self.dev_path = options.dev_path
        if options.test_path:
            self.test_path = options.test_path
        if options.lr:
            self.learning_rate = options.lr
        if options.weight_decay:
            self.weight_decay = options.weight_decay
        if options.hidden_size:
            self.hidden_size = options.hidden_size
        if options.dropout_prob:
            self.dropout_prob = options.dropout_prob
        if options.batch_size:
            self.batch_size = options.batch_size
        if options.epoch:
            self.epochs = options.epoch
        if options.device and options.device != 'cpu':
            self.use_gpu = True
            self.device = options.device
            self.device_id = options.device.split(':')[1]
        if options.show_interval:
            self.show_interval = options.show_interval
        if options.show_times and options.show_interval == 0 and self.show_interval == 0:
            self.show_times = options.show_times

    def show(self):
        # __dict__函数可以获得一个dict，形式为{ 实例的属性 ：属性值 }
        print('\n[  Parameters  ]\n')
        for item in self.__dict__.items():
            print('{}: \'{}\''.format(item[0], item[1]))
        print()


def txt_filter(line):
    return line.replace('-', ' ')

