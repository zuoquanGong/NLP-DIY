# -*- encoding: utf-8 -*-
"""

@Project: SimpleFeeder
@File   : ComponentCategory.py
@Time   : "2019/8/2 10:43
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""

# from Vocabulary import Vocabulary
from SimpleFeeder.Vocabulary import Vocabulary


# =============================================================================
# 一、Component Categories: 组分种类 （Label、Sentence、Dense Feature、Sparse Feature）
# =============================================================================
class LabelGroup:
    group = {}
    vocab_group = {}

    def __init__(self, key, value):
        self.key = key
        if key in self.group.keys():
            self.group[key].append(self)
        else:
            self.group[key] = []
            self.group[key].append(self)
        self.value = value
        self.is_train = False

    @classmethod
    def build_vocab(cls):
        for key in cls.group.keys():
            vocab = Vocabulary(is_label=True)
            vocab.build([unit.value for unit in cls.group[key] if unit.is_train])
            cls.vocab_group[key] = vocab

    @classmethod
    def txt2idx(cls):
        for key in cls.group.keys():
            vocab = cls.vocab_group[key]
            for unit in cls.group[key]:
                for i, word in enumerate(unit.value):
                    unit.value[i] = vocab[word]


class SentenceGroup:
    group = {}
    vocab_group = {}
    ext_vocab_group = {}

    embed_group = {}
    ext_embed_group = {}

    def __init__(self, key, value):
        self.key = key
        if key in self.group.keys():
            self.group[key].append(self)
        else:
            self.group[key] = []
            self.group[key].append(self)
        self.value = value
        self.ext_value = []
        self.is_train = False

    @classmethod
    def build_vocab(cls):
        for key in cls.group.keys():
            vocab = Vocabulary()
            vocab.build([unit.value for unit in cls.group[key] if unit.is_train])
            cls.vocab_group[key] = vocab

    @classmethod
    def txt2idx(cls):
        for key in cls.group.keys():
            vocab = cls.vocab_group[key]
            for unit in cls.group[key]:
                for i, word in enumerate(unit.value):
                    unit.value[i] = vocab[word]

    @classmethod
    def txt2idx_ext(cls):
        for key in cls.group.keys():
            vocab = cls.ext_vocab_group[key]
            for unit in cls.group[key]:
                for i, word in enumerate(unit.value):
                    unit.ext_value.append(vocab[word])


class DenseFeatureGroup:
    group = {}
    vocab_group = {}
    ext_vocab_group = {}

    def __init__(self, key, value):
        self.key = key
        if key in self.group.keys():
            self.group[key].append(self)
        else:
            self.group[key] = []
            self.group[key].append(self)
        self.value = value
        self.ext_value = []
        self.is_train = False

    @classmethod
    def build_vocab(cls):
        for key in cls.group.keys():
            vocab = Vocabulary()
            vocab.build([unit.value for unit in cls.group[key] if unit.is_train])
            cls.vocab_group[key] = vocab

    @classmethod
    def txt2idx(cls):
        for key in cls.group.keys():
            vocab = cls.vocab_group[key]
            for unit in cls.group[key]:
                for i, word in enumerate(unit.value):
                    unit.value[i] = vocab[word]

    @classmethod
    def txt2idx_ext(cls):
        for key in cls.group.keys():
            vocab = cls.ext_vocab_group[key]
            for unit in cls.group[key]:
                for i, word in enumerate(unit.value):
                    unit.ext_value[i].append(vocab[word])


class SparseFeatureGroup:
    group = {}
    vocab_group = {}

    def __init__(self, key, value):
        self.key = key
        if key in self.group.keys():
            self.group[key].append(self)
        else:
            self.group[key] = []
            self.group[key].append(self)
        self.value = value
        self.is_train = False

    @classmethod
    def build_vocab(cls):
        for key in cls.group.keys():
            if key[-4:] == '_num':
                cls.vocab_group[key] = {}
            else:
                vocab = Vocabulary(is_label=True)
                vocab.build([unit.value for unit in cls.group[key] if unit.is_train])
                cls.vocab_group[key] = vocab

    @classmethod
    def txt2idx(cls):
        for key in cls.group.keys():
            vocab = cls.vocab_group[key]
            if vocab == {}:
                for unit in cls.group[key]:
                    for i, word in enumerate(unit.value):
                        unit.value[i] = float(word)
            else:
                for unit in cls.group[key]:
                    for i, word in enumerate(unit.value):
                        unit.value[i] = vocab[word]

