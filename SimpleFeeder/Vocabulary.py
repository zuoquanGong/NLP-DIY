# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : Vocabulary.py
@Time   : "2019/8/2 11:24
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
from typing import Dict, List


# =============================================================================
# 一、Vocabulary
# =============================================================================
class Vocabulary(Dict):
    def __init__(self,
                 is_label=False):
        super(Vocabulary, self).__init__()
        self.is_label = is_label
        if not self.is_label:
            self.padding = '<padding>'
            self.unknown = '<unknown>'
            self.pad_id = 0
            self.unknown_id = 1
            self.id2item = [self.padding,self.unknown]
        else:
            self.id2item = []

    # 字典创建
    def build(self,
              text_list: List[List[str]]):
        for sent in text_list:
            for word in sent:
                if word in self.keys():
                    self[word] += 1
                else:
                    i1 = 2
                    self[word] = i1
        id2freq = sorted(self.items(), key=lambda item: item[1], reverse=True)  # 依据词频进行排序
        self.id2item.extend([item[0] for item in id2freq])
        self.update({item: i for i, item in enumerate(self.id2item)})

    def __getitem__(self,
                    item: str or int) -> int or str:
        if isinstance(item,str):
            if item in self.keys():
                return self.get(item) ##
            else:
                return self.unknown_id
        if isinstance(item,int):
            if item<len(self.id2item):
                return self.id2item[item]
            else:
                exit('Error: data set->vocab->id2item: index out of range.')




