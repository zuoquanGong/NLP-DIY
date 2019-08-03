# -*- encoding: utf-8 -*-
"""

@Project: SimpleFeeder
@File   : Readers.py
@Time   : "2019/8/2 10:47
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
# from Instance import Instance

from SimpleFeeder.Instance import Instance
from typing import List, Tuple
import copy

# SICK 数据集示例
'''
#pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment
4	The young boys are playing outdoors and the man is smiling nearby	There is no boy playing outdoors and there is no man smiling	3.6	CONTRADICTION
24	A person in a black jacket is doing tricks on a motorbike	A skilled person is riding a bicycle on one wheel	3.4	NEUTRAL
'''
def SICK_reader(path: str,
                read_model: List or Tuple,
                separator='\t',
                minicut=' ',
                has_header=True) -> List:
    inst_list = []
    with open(path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()):
            if has_header and i == 0:
                continue
            line = line.strip()
            if line == '':
                continue
            # print(line)
            inst = Instance()
            if isinstance(read_model, list):
                line = line.split(separator)
                # print(line)
                # print(read_model)
                assert len(line) == len(read_model)
                inst.load(line, read_model, minicut=minicut)
            elif isinstance(read_model, tuple):
                inst.load(line, read_model[0], minicut=minicut, separators=read_model[1])
            inst_list.append(inst)
    return inst_list


# POS 序列标注数据示例
'''
上海_NR 浦东_NR 开发_NN 与_CC 法制_NN 建设_NN 同步_VV 
新华社_NN 上海_NR 二月_NT 十日_NT 电_NN （_PU 记者_NN 谢金虎_NR 、_PU 张持坚_NR ）_PU 
'''
def POS_reader(path: str,
                read_model: List,
                separator=' ',
                minicut='_',
                has_header=False) -> List:
    inst_list = []
    with open(path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()):
            if has_header and i == 0:
                continue
            line = line.strip()
            if line == '':
                continue
            # print(line)
            inst = Instance()
            assert isinstance(read_model, list)
            line = line.split(separator)
            # print(line)
            line = [unit.split(minicut) for unit in line]
            # print(line)
            words = [unit[0] for unit in line]
            # print(words)
            tags = [unit[1] for unit in line]
            # print(tags)
            line = [' '.join(words),' '.join(tags)]
            inst.load(line, read_model, minicut=' ')
            inst_list.append(inst)
    return inst_list


# conllu 标注数据示例
'''
# sent_id = train-s111
# text = 喇叭花是温带植物。
1	喇叭花	喇叭花	NOUN	NN	_	4	nsubj	_	SpaceAfter=No
2	是	是	AUX	VC	_	4	cop	_	SpaceAfter=No
3	温带	温带	NOUN	NN	_	4	nmod	_	SpaceAfter=No
4	植物	植物	NOUN	NN	_	18	dep	_	SpaceAfter=No
5	。	。	PUNCT	.	_	18	punct	_	SpaceAfter=No
'''
def conllu_reader(path: str,
                read_model: List,
                separator='\n',
                minicut='\t',
                filter_head_str='#',
                has_header=False) -> List:
    inst_list = []
    components = []
    components_model = []
    for i in range(len(read_model)):
        components_model.append([])
    with open(path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.read().split('\n\n')):
            if has_header and i == 0:
                continue
            if line == '':
                continue

            inst = Instance()
            components = copy.deepcopy(components_model)
            for unit_line in line.split(separator):
                if unit_line[0] == filter_head_str:
                    continue
                unit_line = unit_line.strip()
                if unit_line == '':
                    continue
                for i, unit_component in enumerate(unit_line.split(minicut)):
                    components[i].append(unit_component)
            line = [' '.join(component) for component in components]
            # print(line)

            inst.load(line, read_model, minicut=' ')
            inst_list.append(inst)
    return inst_list

if __name__ == "__main__":
    '''
    # sent_id = train-s111
    # text = 喇叭花是温带植物。
    1	喇叭花	喇叭花	NOUN	NN	_	4	nsubj	_	SpaceAfter=No
    2	是	是	AUX	VC	_	4	cop	_	SpaceAfter=No
    3	温带	温带	NOUN	NN	_	4	nmod	_	SpaceAfter=No
    4	植物	植物	NOUN	NN	_	18	dep	_	SpaceAfter=No
    5	。	。	PUNCT	.	_	18	punct	_	SpaceAfter=No
    '''
    path = 'G:\桌面\语料搜集\conllu.txt'
    read_model = ['', 'sent:A', '', 'dfeat:pos', '', '', '', 'label:dep', '', '']
    inst_list = conllu_reader(path, read_model)
    print(inst_list[0])
