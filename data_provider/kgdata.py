import argparse
import os
import pytorch_lightning as pl
import torch
from collections import defaultdict as ddict
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
from collections import deque, defaultdict
from math import sqrt
from collections import Counter
import dgl
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KGData(object):
    """Data preprocessing of kg data.

    Attributes:
        args: Some pre-set parameters, such as dataset path, etc. 
        ent2id: Encoding the entity in triples, type: dict.
        rel2id: Encoding the relation in triples, type: dict.
        id2ent: Decoding the entity in triples, type: dict.
        id2rel: Decoding the realtion in triples, type: dict.
        train_triples: Record the triples for training, type: list.
        valid_triples: Record the triples for validation, type: list.
        test_triples: Record the triples for testing, type: list.
        all_true_triples: Record all triples including train,valid and test, type: list.
        hr2t_train: Record the tail corresponding to the same head and relation, type: defaultdict(class:set).
        rt2h_train: Record the head corresponding to the same tail and relation, type: defaultdict(class:set).
        h2rt_train: Record the tail, relation corresponding to the same head, type: defaultdict(class:set).
        t2rh_train: Record the head, realtion corresponding to the same tail, type: defaultdict(class:set).
    """

    def __init__(self, args):
        self.args = args

        #  基础部分
        self.ent2id = {}
        self.rel2id = {}
        # predictor需要
        self.id2ent = {}
        self.id2rel = {}
        # 存放三元组的id
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()

        # 存放头实体和尾实体的关系
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.h2rt_train = ddict(set)
        self.t2rh_train = ddict(set)
        self.htr_train=ddict(set)
        # 初始化
        self.get_id()
        self.get_triples_id()
        self.dgl_graph = self.build_dgl_graph()
        # 存储一下region的onehop邻居
        self.region_neighbors=ddict(list)
        self.get_region_neighbors()
    
    def build_dgl_graph(self):
        """根据 train_triples 构建 DGL 图并在边上存储 relation"""
        src, dst, rel_data = [], [], []
        for h, r, t in self.train_triples:
            src.append(h)
            dst.append(t)
            rel_data.append(r)
        g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=self.args.num_ent)
        g.edata['rel'] = torch.tensor(rel_data)
        return g

    def get_l_hop_triples_dgl(self, g, start_entity, L):
        """在有向图中进行双向BFS，获取出度与入度"""
        visited = set([start_entity])
        queue = [(start_entity, 0)]
        
        while queue:
            cur_ent, depth = queue.pop(0)
            if depth < L:
                # 包含出度和入度
                out_neighbors = g.successors(cur_ent).tolist()
                in_neighbors = g.predecessors(cur_ent).tolist()
                for n in out_neighbors + in_neighbors:
                    if n not in visited:
                        visited.add(n)
                        queue.append((n, depth + 1))
    
        # 对访问到的节点构建子图
        node_ids = list(visited)
        sg = g.subgraph(node_ids)
        src, dst = sg.edges()
        rels = sg.edata['rel']
        
        # 恢复原始节点ID并组装三元组
        triples = []
        nid_map = sg.ndata[dgl.NID].tolist()
        for i in range(sg.num_edges()):
            h = nid_map[src[i]]
            t = nid_map[dst[i]]
            r = rels[i].item()
            triples.append((h, r, t))
        return triples
    

    def triple2text(self, triples):
        """将三元组列表转换为文本"""
        text = []
        for h, r, t in triples:
            text.append(f'{self.id2ent[h]} {self.id2rel[r]} {self.id2ent[t]}')
        text='\t'.join(text)
        return text
    
    def text2tokens(self, text, tokenizer):
        """将文本转换为token"""
        tokens = tokenizer(text, truncation=True, padding=True)
        return tokens
    
    def get_id(self):
        """Get entity/relation id, and entity/relation number.

        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.id2ent: id to Entity.
            self.id2rel: id to Relation.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        
        with open(os.path.join(self.args.data_path, "entities.dict")) as fin:
            for line in fin:
                eid, entity = line.strip().split("\t")
                self.ent2id[entity] = int(eid)
                self.id2ent[int(eid)] = entity

        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation] = int(rid)
                self.id2rel[int(rid)] = relation

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)

    def get_triples_id(self):
        """Get triples id, save in the format of (h, r, t).

        Update:
            self.train_triples: Train dataset triples id.
            self.valid_triples: Valid dataset triples id.
            self.test_triples: Test dataset triples id.
        """
        
        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_hr2t_rt2h_from_train(self):
        """Get the set of hr2t and rt2h from train dataset, the data type is numpy.

        Update:
            self.hr2t_train: The set of hr2t.
            self.rt2h_train: The set of rt2h.
        """
        
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))

    @staticmethod
    def count_frequency(triples, start=4):
        '''Get frequency of a partial triple like (head, relation) or (relation, tail).
        
        The frequency will be used for subsampling like word2vec.
        
        Args:
            triples: Sampled triples.
            start: Initial count number.

        Returns:
            count: Record the number of (head, relation).
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    def get_htr_from_train(self):
        '''
        得到头实体和尾实体的关系
        '''    
        for h,r,t in self.train_triples:
            self.htr_train[(h,t)].add(r)
        for h,r,t in self.all_true_triples:
            self.htr[(h,t)].add(r)
            

    def get_h2rt_t2hr_from_train(self):
        """Get the set of h2rt and t2hr from train dataset, the data type is numpy.

        Update:
            self.h2rt_train: The set of h2rt.
            self.t2rh_train: The set of t2hr.
        """
        for h, r, t in self.train_triples:
            self.h2rt_train[h].add((r, t))
            self.t2rh_train[t].add((r, h))
        for h in self.h2rt_train:
            self.h2rt_train[h] = np.array(list(self.h2rt_train[h]))
        for t in self.t2rh_train:
            self.t2rh_train[t] = np.array(list(self.t2rh_train[t]))
     
    def get_region_neighbors(self):
        '''得到region的一跳邻居'''
        for ent in self.ent2id:
            if ent.startswith('region'):
                self.region_neighbors[ent]=self.get_l_hop_triples_dgl(self.dgl_graph,self.ent2id[ent],1)


def zh2en(file):
    # 读取文件内容
    result=defaultdict()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    result={i.strip().split(',')[0]:i.strip().split(',')[1] for i in lines}
    
    return result   
def jifengattr(file,entity,triples):
    # 读取文件内容
    result=defaultdict()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines=lines[1:]
    for i in lines:
        tmp=i.strip().split(',')
        tmp[1]=tmp[1].strip().replace(' ','_')
        tmp[2]=tmp[2].strip().replace(' ','_')
        tmp[3]=tmp[3].strip().replace(' ','_')
        entity.add(tmp[1])
        entity.add(tmp[2])
        entity.add(tmp[3])
        triples.append((tmp[1],'moonsoon_start',tmp[2]))
        triples.append((tmp[1],'moonsoon_end',tmp[3]))
    
    return result        
def convert_data_kg(grid_data,currentfile=None,monsoonfile=None):
    # # # 将数据转换为三元组
    with open (grid_data) as f:
        data = f.readlines()
    triples = []
    entities = set()  
    relations = set()
    relations.add('belongs_to')
    relations.add('is_at')
    relations.add('has_current')
    relations.add('monsoon_impact')
    relations.add('moonsoon_start')
    relations.add('moonsoon_end')
    with open (currentfile) as f:
        currentdata = f.readlines()
    with open (monsoonfile) as f:
        monsoondata = f.readlines()
    current_zh2en = zh2en(currentfile.split('_influence.csv')[0]+'.csv')
    currentdata = currentdata[1:]
    
    monsoondata_zhen = zh2en(monsoonfile.split('_influence.csv')[0]+'.csv')
    monsoondata = monsoondata[1:]
    
    jifengattr(monsoonfile.split('_influence.csv')[0]+'.csv',entities,triples)
    
    for i in range(len(currentdata)):
        tmp = currentdata[i].strip().split(',')
        if len(tmp)==4:
            batch_id,current = tmp[:2]
            if current in current_zh2en:
                current = current_zh2en[current]
            current=current.strip().replace(' ','_')
            entities.add(current)
            entities.add('region_'+batch_id)
            triples.append(('region_'+batch_id,'has_current',current))
            
        else:
            assert False
          
    for i in range(len(monsoondata)):
        tmp = monsoondata[i].strip().split(',')
        if len(tmp)==2:
            batch_id,monsoon = tmp[:2]
            monsoon = monsoondata_zhen[monsoon]
            monsoon=monsoon.strip().replace(' ','_')
            entities.add(monsoon)
            entities.add('region_'+batch_id)
            triples.append(('region_'+batch_id,'monsoon_impact',monsoon))
        else:
            assert False
            
    for line in data:
        itme= line.strip().split(',')
        if len(itme)==5:
            batch_id,ns,we,ocean,qihou = itme
        elif len(itme)==6:
            batch_id,ns,we,ocean,qihou,_ = itme
            # current = current.strip().replace(' ','_')
            # triples.append(('region_'+batch_id,'has_current',current))
            # entities.add(current)
        # elif len(itme)==7:
        #     batch_id,ns,we,ocean,qihou,current,temperature = itme
        #     current = current.strip().replace(' ','_')+'_&_'+temperature.strip().replace(' ','_')
        #     triples.append(('region_'+batch_id,'has_current',current))
        #     entities.add(current)
        else:
            assert False   
        ocean=ocean.strip().replace(' ','_')
        qihou=qihou.strip().replace(' ','_')
        triples.append(('region_'+batch_id,'belongs_to',ocean))
        triples.append(('region_'+batch_id,'is_at',qihou))
        entities.add('region_'+batch_id)
        entities.add(ocean)
        entities.add(qihou)
        if not ocean or not qihou or not batch_id:
            import ipdb;ipdb.set_trace()
    with open('entities.dict','w') as f:
        for i,entity in enumerate(entities):
            f.write(f'{i}\t{entity}\n')
    with open('relations.dict','w') as f:
        for i,relation in enumerate(relations):
            f.write(f'{i}\t{relation}\n')
    with open('train.txt','w') as f:
        for h,r,t in triples:
            f.write(f'{h}\t{r}\t{t}\n')
    return triples
        # 将数据转换为三元组
    # with open (grid_data) as f:
    #     data = f.readlines()
    # triples = []
    # entities = set()  
    # relations = set()
    # relations.add('belongs_to')
    # relations.add('is_at')
    # relations.add('has_current')
    # index = 0
    # for line in data:
    #     itme= line.strip().split(',')
    #     if len(itme)==4:
    #         ns,we,ocean,qihou = itme
    #     elif len(itme)==5:
    #         ns,we,ocean,qihou,current = itme
    #         current = current.strip().replace(' ','_')
    #         triples.append(('region_'+str(index),'has_current',current))
    #         entities.add(current)
    #     elif len(itme)==6:
    #         ns,we,ocean,qihou,current,temperature = itme
    #         current = current.strip().replace(' ','_')+'_&_'+temperature.strip().replace(' ','_')
    #         triples.append(('region_'+str(index),'has_current',current))
    #         entities.add(current)
    #     else:
    #         assert False   
    #     ocean=ocean.strip().replace(' ','_')
    #     qihou=qihou.strip().replace(' ','_')
    #     triples.append(('region_'+str(index),'belongs_to',ocean))
    #     triples.append(('region_'+str(index),'is_at',qihou))
    #     entities.add('region_'+str(index))
    #     entities.add(ocean)
    #     entities.add(qihou)
    #     if not ocean or not qihou:
    #         import ipdb;ipdb.set_trace()
    #     index+=1
    # with open('entities.dict','w') as f:
    #     for i,entity in enumerate(entities):
    #         f.write(f'{i}\t{entity}\n')
    # with open('relations.dict','w') as f:
    #     for i,relation in enumerate(relations):
    #         f.write(f'{i}\t{relation}\n')
    # with open('train.txt','w') as f:
    #     for h,r,t in triples:
    #         f.write(f'{h}\t{r}\t{t}\n')
      # # 将数据转换为三元组
    # with open (grid_data) as f:
    #     data = f.readlines()
    # triples = []
    # entities = set()  
    # relations = set()
    # relations.add('belongs_to')
    # relations.add('is_at')
    # relations.add('has_current')
    # lines=[]
    # for line in data:
    #     itme= line.strip().split(',')
    #     if len(itme)==5:
    #         itemtxt=','.join(itme)
    #         itemtxt+=', '
    #     elif len(itme)==6:
    #         itemtxt=','.join(itme)
    #     else:
    #         assert False
    #     itemtxt+='\n'
    #     lines.append(itemtxt)
    # with open(grid_data+'v','w') as f:
    #     for line in lines:
    #         f.write(line)
    # return 1

if __name__ == '__main__':
    # 将字典转换为ArgumentParser对象
    def dict_to_args(args_dict):
        parser = argparse.ArgumentParser(description="KG Model Arguments")
        for key, value in args_dict.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        return parser.parse_args([])
    
     # 定义参数字典
    args_dict = {
        'data_path': '/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/Time-LLM/data_provider/',
        'num_ent':0,
        'num_rel':0,
        'pretrain_path':'./',
        'pretrain_emb':True,
        
    }
    # parser = argparse.ArgumentParser(description="KG Model Arguments")
    # parser.add_argument('--data_path', type=str, default='/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/Time-LLM/data_provider/')
    # parser.add_argument('--num_ent', type=int, default=0)
    # parser.add_argument('--num_rel', type=int, default=0)
    # parser.add_argument('--pretrain_path', type=str, default='./')
    # parser.add_argument('--pretrain_emb', type=bool, default=True)
    # args = parser.parse_args()
    
    
    # args = dict_to_args(args_dict)
    # kgdata = KGData(args)
    # triples=kgdata.get_l_hop_triples_dgl(kgdata.dgl_graph, 0, 1)
    # print(triples)
    # print(kgdata.triple2text(triples))
    
    # 生成数据 运行一次即可
    triples = convert_data_kg('/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/KG-LLM/input_data_for_model/processed_grid_centers_currents_1.txt',
                              '/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/KG-LLM/input_data_for_model/currents_influence.csv',
                              '/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/KG-LLM/input_data_for_model/monsoon_influence.csv')
    