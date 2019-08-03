# -*- encoding: utf-8 -*-
"""

@Project: SimpleFeeder
@File   : SimpleFeeder.py
@Time   : "2019/7/30 21:00
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
# 本文件中定义的诸类，用于实现数据集无关化
# 即只需提供数据集的组分、类别和文本格式就可以自动加载数据，
# 并可以经过自动处理，直接将数据提供给模型
# 组分表示例：{'label:0','','sent:0','sent:1','sfeat:0',''}
# 实例句子：“相关   ### 丁磊创建了网易公司。  丁磊是网易公司创始人。 丁磊  -1”
# 标注说明：label:0 ''     sent:0             sent:1       sfeat:0 ''
# '' 部分将被省略掉，sfeat 代表 sparse feature，dfeat 代表 dense feature

from typing import Dict
import torch.nn as nn
import numpy as np
from torch import LongTensor, ByteTensor, FloatTensor
from torch.autograd import Variable
import re

# from ComponentCategory import LabelGroup, SentenceGroup, DenseFeatureGroup, SparseFeatureGroup
# # from Readers import SICK_reader, POS_reader, conllu_reader
# from Readers import *
# from Vocabulary import Vocabulary

from SimpleFeeder.ComponentCategory import LabelGroup, SentenceGroup, DenseFeatureGroup, SparseFeatureGroup
# from Readers import SICK_reader, POS_reader, conllu_reader
from SimpleFeeder.Readers import *
from SimpleFeeder.Vocabulary import Vocabulary


# =============================================================================
#  InstanceList
# =============================================================================
# 所有Instance的容器，相当于一个数据集
# 可以在其中完成常规数据处理流程

class InstanceList(List):
    has_sent_pretrain = False
    embed_dim = 0
    batch_size = 0
    is_POS = False

    def __init__(self,
                 is_train=False):
        super(InstanceList, self).__init__()
        self.is_train = is_train
        self.indices = []
        self.batch_num = 0

    # 1.从文件中加载数据
    def read(self,
             path: str,
             read_model: List[str] or str,
             reader,
             is_POS=False,
             separator='\t',
             minicut=' ',
             no_fix=False):
        if isinstance(read_model, str):
            p_component = re.compile('\[.*?\]')
            p_separator = re.compile('\].*?\[')
            separators = [s[1:-1] for s in p_separator.findall(read_model)]
            components = [c[1:-1] for c in p_component.findall(read_model)]
            # print(components)
            # print(separators)
            read_model = (components, separators)
        if reader == POS_reader or is_POS or reader == conllu_reader:
            self.is_POS = True
        self.clear()
        if no_fix:
            self.extend(reader(path, read_model))
        else:
            self.extend(reader(path, read_model, separator=separator, minicut=minicut))
        self.indices = list(range(len(self)))
        if self.is_train:
            self.set_train()
        print("Loading is finished -- {} insts ".format(len(self)))

    # 2.建立词表
    def build_vocab(self):
        if not self.is_train:
            return
        LabelGroup.build_vocab()
        SentenceGroup.build_vocab()
        DenseFeatureGroup.build_vocab()
        SparseFeatureGroup.build_vocab()

    # 3.将文本转化为索引
    def txt2idx(self):
        if not self.is_train:
            print("warning: can't execute this function for this dataset, please set it trainning（set_train）.")
            return
        LabelGroup.txt2idx()
        SentenceGroup.txt2idx()
        DenseFeatureGroup.txt2idx()
        SparseFeatureGroup.txt2idx()

    # 4。加载词嵌入
    def load_sent_pretrain(self,
             path: str,
             key='sent'):
        if not self.is_train:
            print('warning: non-train data set has no such function: "load_sent_pretrain".')
            return
        self.has_sent_pretrain = True
        idx = 0
        ext_vocab = Vocabulary()
        embed_dim = -1
        word2vec = []
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                line.strip()
                line = line.split()
                if len(line) < 2:
                    continue

                # 测试用语
                # if idx > 2000:  ####
                #     break

                word, vec = line[0],line[1:]
                if embed_dim == -1:  # 判断vec维数是否合理
                    embed_dim = len(vec)
                else:
                    assert embed_dim == len(vec), '词向量维度不一致'

                ext_vocab[word] = idx
                word2vec.append(np.array(vec))
                idx += 1

        oov_list = [word for word in SentenceGroup.vocab_group[key] if word not in ext_vocab]
        oov_count = len(oov_list)
        print('oov list: ', oov_list)
        print('oov count: ', oov_count)
        print('oov ratio: %.3f %%' % (100*oov_count/len(SentenceGroup.vocab_group[key])))
        self.embed_dim = embed_dim
        pretrain_weight = np.zeros((len(ext_vocab), embed_dim))
        for i, vec in enumerate(word2vec):
            pretrain_weight[i] = vec

        ext_embedding = nn.Embedding(len(ext_vocab),embed_dim)
        ext_embedding.weight.data.copy_(FloatTensor(pretrain_weight))
        ext_embedding.weight.requires_grad = False
        del word2vec
        del pretrain_weight
        SentenceGroup.ext_vocab_group[key] = ext_vocab
        SentenceGroup.ext_embed_group[key] = ext_embedding
        print(SentenceGroup.ext_vocab_group[key])
        SentenceGroup.txt2idx_ext()

        print('Extra_embedding is done.')

    # 5.建立词嵌入
    def build_sent_pretrain(self,
                       embed_dim: int,
                       key='sent'):
        if not self.has_sent_pretrain:
            self.embed_dim = embed_dim
        embed = nn.Embedding(len(SentenceGroup.vocab_group[key]), self.embed_dim)
        embed.weight.requires_grad = True
        SentenceGroup.embed_group[key] = embed

    # 6.batch生成器
    def batch_generator(self,
                        batch_size: int,
                        use_gpu=False,
                        device='cpu'):
        self.batch_size = batch_size
        self.batch_num = len(self) // batch_size

        class BatchData:
            size = 0

            def __init__(self,
                         batch_tensor_dict,
                         batch_tensor_dict_ext):
                self.batch_tensor_dict = batch_tensor_dict
                self.batch_tensor_dict_ext = batch_tensor_dict_ext

        BatchData.size = self.batch_size
        for batch in range(self.batch_num):
            batch_indices = self.indices[batch * batch_size:(batch + 1) * batch_size]

            batch_insts = [self[idx] for idx in batch_indices]
            batch_tensor_dict = {}
            batch_tensor_dict_ext = {}
            if not use_gpu:
                device = 'cpu'

            for attr_name in self[0].attr.keys():
                is_label = False
                is_sparse = False
                if 'sfeat' in self[0].attr[attr_name].key:
                    is_sparse = True
                elif 'label' in self[0].attr[attr_name].key:
                    is_label = True
                attr_list = [inst.attr[attr_name] for inst in batch_insts]
                # print(attr_name)
                batch_tensor_dict[attr_name] = self.list2tensor(attr_list, is_sparse=is_sparse, is_label=is_label, device=device)
                if self.has_sent_pretrain and 'sent' in self[0].attr[attr_name].key:
                    batch_tensor_dict_ext[attr_name] = self.list2tensor(attr_list, is_ext=True, device=device)

            yield BatchData(batch_tensor_dict, batch_tensor_dict_ext)  # 生成批数据类实例，通过迭代调用

    ###############
    # 7.设置当前数据集为训练集
    def set_train(self):
        self.is_train = True
        for inst in self:
            for attr in inst.attr.values():
                attr.is_train = True

    # 8.输出标签个数
    def label_size(self,
                   key: str) -> int:
        if key not in LabelGroup.vocab_group.keys():
            print('warning: "{}" is a wrong key, no such label.'.format(key))
            return 0
        return len(LabelGroup.vocab_group[key])

    # 9.数据集内部数据洗牌
    def shuffle(self):
        np.random.shuffle(self.indices)

    # 10.将list转为tensor，用于batch_generator
    def list2tensor(self,
                    in_list: List,
                    is_label=False,
                    is_ext=False,
                    is_sparse=False,
                    device='cpu'):
        if is_sparse:
            tensor = Variable(FloatTensor(len(in_list)).zero_())
            for s, sparse in enumerate(in_list):
                # print(sparse.value)
                tensor[s] = sparse.value[0]
            # print(tensor)
            return tensor
        elif not is_label:
            if not is_ext:
                in_list = [attr.value for attr in in_list]
            else:
                in_list = [attr.ext_value for attr in in_list]
            max_len = max([len(unit) for unit in in_list])
            tensor = Variable(LongTensor(len(in_list), max_len).zero_())
            mask = Variable(ByteTensor(len(in_list), max_len).zero_())

            for idx, unit in enumerate(in_list):
                unit_len = len(unit)
                tensor[idx, :unit_len] = LongTensor(unit)
                mask[idx, :unit_len].fill_(1)
            tensor.to(device=device)
            mask.to(device=device)
            return tensor, mask
        elif self.is_POS:
            in_list = [attr.value for attr in in_list]
            max_len = max([len(unit) for unit in in_list])
            tensor = Variable(LongTensor(len(in_list), max_len).zero_())
            mask = Variable(ByteTensor(len(in_list), max_len).zero_())

            for idx, unit in enumerate(in_list):
                unit_len = len(unit)
                tensor[idx, :unit_len] = LongTensor(unit)
                mask[idx, :unit_len].fill_(1)
            tensor.to(device=device)
            mask.to(device=device)
            return tensor, mask
        else:
            label_size = len(LabelGroup.vocab_group[in_list[0].key])
            tensor = Variable(LongTensor(len(in_list), label_size).zero_())
            for idx, label in enumerate(in_list):
                for lid in range(label_size):
                    if lid == label.value[0]:
                        tensor[idx][lid] = 1
            tensor.to(device=device)
            return tensor

    def get_pretrain(self) -> Dict:
        embed = {}
        embed.update(SentenceGroup.embed_group)
        # embed.update(DenseFeatureGroup.embed_group)
        # embed.update(SparseFeatureGroup.embed_group)
        return embed

    def get_pretrain_ext(self):
        embed = {}
        embed.update(SentenceGroup.ext_embed_group)
        # embed.update(DenseFeatureGroup.embed_group)
        # embed.update(SparseFeatureGroup.embed_group)
        return embed
