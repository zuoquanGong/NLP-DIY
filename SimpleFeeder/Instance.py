# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : Instance.py
@Time   : "2019/8/2 11:29
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
from typing import List, Tuple
from ComponentCategory import LabelGroup, SentenceGroup, DenseFeatureGroup, SparseFeatureGroup


# =============================================================================
# 一、Instance
# =============================================================================
class Instance:
    def __init__(self):
        self.attr = {}

    def load(self,
             line: List[str] or Tuple,
             read_model: List,
             separators='',
             minicut=' '):
        components = read_model
        if separators != '':
            # 对应着str类型的read_model
            line_list = []
            for separator in separators:
                line = line.split(separator, 1)
                if len(line) < 2:
                    if len(line_list) < len(components):
                        line_list.append(line[0])
                    break
                line_list.append(line[0])
                line = line[1]
                if line == '':
                    break
            if len(line_list) < len(components) and line != '':
                line_list.append(line)
            # print(line_list)
            assert len(line_list) == len(components)
        else:  # 对应着list类型的read_model
            line_list = line

        for component, string in zip(components, line_list):
            # print(read_model)
            # print(line)
            if component == '':
                continue
            component_split = component.split(':')
            assert len(component_split) == 2
            if 'label' in component_split[0]:
                self.attr[component] = LabelGroup(component_split[0], string.split(minicut))
            elif 'sent' in component_split[0]:
                self.attr[component] = SentenceGroup(component_split[0], string.split(minicut))
            elif 'dfeat' in component_split[0]:
                self.attr[component] = DenseFeatureGroup(component_split[0], string.split(minicut))
            elif 'sfeat' in component_split[0]:
                self.attr[component] = SparseFeatureGroup(component_split[0], string.split(minicut))
            else:
                continue

    def __str__(self) -> str:
        string = ''
        for attr_item in self.attr.items():
            string += '"'+attr_item[0].__str__()+'".'
            string += attr_item[1].value.__str__()
            string += '\n'
            if 'sent' in attr_item[1].key:
                if attr_item[1].ext_value:
                    string += '"'+attr_item[0].__str__()+'"-ext.'
                    string += attr_item[1].ext_value.__str__()
                    string += '\n'
        return string