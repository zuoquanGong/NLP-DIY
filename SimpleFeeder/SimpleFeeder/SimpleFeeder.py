# -*- encoding: utf-8 -*-
"""

@Project: ESIM
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

from typing import List
import torch.nn as nn
import numpy as np
from torch import LongTensor, ByteTensor, FloatTensor
from torch.autograd import Variable
import re

from ComponentCategory import LabelGroup, SentenceGroup, DenseFeatureGroup, SparseFeatureGroup
# from Readers import SICK_reader, POS_reader, conllu_reader
from Readers import *
from Vocabulary import Vocabulary


# =============================================================================
# 一、InstanceList
# =============================================================================
# 所有Instance的容器，相当于一个数据集
# 可以在其中完成常规数据处理流程

class InstanceList(List):
    has_sent_pretrain = False
    embed_dim = 0
    batch_size = 0
    batch_num = 0
    is_POS = False

    def __init__(self,
                 is_train=False):
        super(InstanceList, self).__init__()
        self.is_train = is_train
        self.indices = []

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
    def build_pretrain(self,
                       embed_dim: int,
                       key='sent'):
        if not self.has_sent_pretrain:
            self.embed_dim=embed_dim
        embed = nn.Embedding(len(SentenceGroup.vocab_group[key]), self.embed_dim)
        embed.weight.requires_grad=True
        SentenceGroup.embed_group[key] = embed

    # 6.batch生成器
    def batch_generator(self,
                        batch_size: int,
                        use_gpu=False,
                        device='cpu'):
        self.batch_size = batch_size
        self.batch_num = len(self) // batch_size

        class BatchData:
            def __init__(self,
                         batch_tensor_dict,
                         batch_tensor_dict_ext):
                self.batch_tensor_dict = batch_tensor_dict
                self.batch_tensor_dict_ext = batch_tensor_dict_ext

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


# =============================================================================
# 二、Tests and Examples
# =============================================================================
# 使用流程：
# 首先，定义数据路径（path），数据读取格式（read_model），读取方法（reader），组分间分隔符和成分内分隔符（separator，minicut)
# 注：例如 "A-component \t B-component \t C-component"，A、B、C组分间的分隔符为“\t”
# 注：C-component 假设是句子 “W1<space>W2<space>W3<space>W4” ，则 minicut 为“<space>”
# 然后，进入流程，依次调用 InstanceList的 read、set_train、build_vocab、build_pretrain、txt2idx、batch_generator
# batch_generator的输出即可以满足模型输入

if __name__ == "__main__":
    train = InstanceList(is_train=True)

    '''
    #pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment
    4	The young boys are playing outdoors and the man is smiling nearby	There is no boy playing outdoors and there is no man smiling	3.6	CONTRADICTION
    24	A person in a black jacket is doing tricks on a motorbike	A skilled person is riding a bicycle on one wheel	3.4	NEUTRAL
    '''
    # path = 'G:\桌面\语料搜集\SICK.txt'
    # read_model = ['', 'sent:A', 'sent:B', '', 'label:entail']
    # reader = SICK_reader
    # train.read(path, read_model, reader=reader, separator='\t', minicut=' ')

    '''
    #has_relation   sentence_A  sentence_B  sparse_feature_dist
    1	房间 很 大 ， 用品 设施 齐全 ， 还有 麻将 室 和 自动 麻将桌 ， 虽然 没 用上 吧 ； 工作 人员 服务 态度 不错 ； 因为 挨着 高架桥 ， 晚上 会 有 车辆 穿行 的 声音 ， 不过 总体 来说 还 不错 ， 性价比 蛮 高 的 。	性价比 蛮 高的	12
    0	钟楼 附近 ， 观景 购物 很 方便 ， 去 哪儿 网 订 的 房间 ， 并 不 贵 ， 398 元 ， 房间 空间 大 ， 有 近 40 平米 ， 淋浴 不错 。	房间 观景 购物 很 方便	77
    '''
    # path = 'G:\桌面\语料搜集\opinion_relation.txt'
    # read_model = ['label:rel', 'sent:A', 'sent:B', 'sfeat_num:dist']
    # reader = SICK_reader
    # train.read(path, read_model, reader=reader, separator='\t', minicut=' ')

    '''
    #has_relation   sentence_A  sentence_B  sparse_feature_dist sparse_feature_polar
    1	环境 很 好 ， 在 夫子庙 景区 中心 ， 但 房间 设施 陈旧 ， 特别是 卫生间 座便 目不忍睹 ， 应该 换了 。	环境 很好	11 1
    0	环境 很 好 ， 在 夫子庙 景区 中心 ， 但 房间 设施 陈旧 ， 特别是 卫生间 座便 目不忍睹 ， 应该 换了 。	环境 陈旧	87 0
    '''
    # path = 'G:\桌面\语料搜集\opinion_relation2.txt'
    # read_model = '[label:rel]	[sent:A]	[sent:B]	[sfeat_num:dist] [sfeat:polar]'
    # reader = SICK_reader
    # train.read(path, read_model, reader=reader, separator='\t', minicut=' ')

    '''
    上海_NR 浦东_NR 开发_NN 与_CC 法制_NN 建设_NN 同步_VV 
    新华社_NN 上海_NR 二月_NT 十日_NT 电_NN （_PU 记者_NN 谢金虎_NR 、_PU 张持坚_NR ）_PU 
    '''
    # path = 'G:\桌面\语料搜集\pos.txt'
    # read_model = ['sent:word', 'label:tag']
    # reader = POS_reader
    # train.read(path, read_model, reader=reader, separator=' ', minicut='_')

    '''
        # sent_id = train-s111
        # text = 喇叭花是温带植物。
        1	喇叭花	喇叭花	NOUN	NN	_	4	nsubj	_	SpaceAfter=No
        2	是	是	AUX	VC	_	4	cop	_	SpaceAfter=No
        3	温带	温带	NOUN	NN	_	4	nmod	_	SpaceAfter=No
        4	植物	植物	NOUN	NN	_	18	dep	_	SpaceAfter=No
        5	。	。	PUNCT	.	_	18	punct	_	SpaceAfter=No
        '''
    # path = 'G:\桌面\语料搜集\conllu.txt'
    # read_model = ['', 'sent:A', '', 'dfeat:pos', '', '', '', 'label:dep', '', '']
    # reader = conllu_reader
    # train.read(path, read_model, reader, no_fix=True)

    print(train[0])
    print(train[1])
    print(train[2])
    train.set_train()  # 给予数据集操作权限
    train.build_vocab()  # 建立词表
    print(LabelGroup.vocab_group)
    print(DenseFeatureGroup.vocab_group)
    # print(SparseFeatureGroup.vocab_group)
    # print(SparseFeatureGroup.vocab_group['sfeat_num'])
    # print(SparseFeatureGroup.vocab_group['sfeat'])
    train.build_pretrain(2)  # 建立词嵌入——Embedding
    train.txt2idx()  # 将text数据映射成index数据
    print(train[0])
    print(train[1])
    print(train[2])
    for i, batch in enumerate(train.batch_generator(32)):  # batch 生成器
        if i > 0:
            break
        print(batch.__dict__.items())
