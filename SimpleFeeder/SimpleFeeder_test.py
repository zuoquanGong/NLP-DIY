# -*- encoding: utf-8 -*-
"""

@Project: SimpleFeeder
@File   : SimpleFeeder.py
@Time   : "2019/7/30 21:00
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""

from SimpleFeeder.SimpleFeeder import InstanceList
from SimpleFeeder.ComponentCategory import SentenceGroup, LabelGroup, SparseFeatureGroup, DenseFeatureGroup
from SimpleFeeder.Readers import SICK_reader, POS_reader, conllu_reader

# =============================================================================
# Tests and Examples
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
    path = 'G:\桌面\语料搜集\opinion_relation2.txt'
    read_model = '[label:rel]	[sent:A]	[sent:B]	[sfeat_num:dist] [sfeat:polar]'
    reader = SICK_reader
    train.read(path, read_model, reader=reader, separator='\t', minicut=' ')

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
    train.build_sent_pretrain(2)  # 建立词嵌入——Embedding
    train.txt2idx()  # 将text数据映射成index数据
    print(train[0])
    print(train[1])
    print(train[2])
    for i, batch in enumerate(train.batch_generator(32)):  # batch 生成器
        if i > 0:
            break
        print(batch.__dict__.items())