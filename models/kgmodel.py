import argparse
import pytorch_lightning as pl
import torch
from collections import defaultdict as ddict
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
from collections import deque, defaultdict
from math import sqrt
from collections import Counter

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# adapter for llm 用于将图嵌入映射到LLM期望的输入维度
class GraphEmbeddingAdapter(nn.Module):
    def __init__(self, graph_embed_dim, llm_embed_dim):
        super().__init__()
        # 将图嵌入维度投影到LLM输入期望的embedding维度
        self.projection = nn.Linear(graph_embed_dim, llm_embed_dim)

    def forward(self, graph_embeddings):
        """
        graph_embeddings: [batch_size, graph_embed_dim]
        返回 [batch_size, llm_embed_dim]
        """
        return self.projection(graph_embeddings)

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
# kg graph model for llm
class KGModel(nn.Module):

    def __init__(self, args):
        super(KGModel, self).__init__()
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()
        
        # self.ffn = FFN(input_dim=2 * self.args.emb_dim, hidden_dim=self.args.hidden_dim)
        self.adapter=GraphEmbeddingAdapter(self.args.emb_dim,self.args.hidden_dim)
    def init_emb(self):
        """Initialize the entity and relation embeddings
        """
        if self.args.pretrain_emb:
            # 有一个包含 pretrain_emb 的 state_dict:
            # with torch.serialization.safe_globals([torch.nn.Embedding]):
            data = torch.load(self.args.pretrain_path, map_location='cpu')
            
            # 提取张量
            self.ent_emb = data['entity_embedding'].to(self.args.device) 
            self.rel_emb = data['relation_embedding'].to(self.args.device)

            # 冻结梯度
            self.ent_emb.requires_grad_(False)
            self.rel_emb.requires_grad_(False)
            
        else:
            self.epsilon = 2.0
            self.margin = nn.Parameter(
                torch.Tensor([self.args.margin]), 
                requires_grad=False
            )
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
                requires_grad=False
            )

            self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
            self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
            nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())        
            
    def tri2emb(self, triples):
        """Get embedding of triples.
        
        This function get the embeddings of head, relation, and tail
        respectively. each embedding has three dimensions.

        Args:
            triples (tensor): This tensor save triples id, which dimension is 
                [triples number, 3].
            negs (tensor, optional): This tenosr store the id of the entity to 
                be replaced, which has one dimension. when negs is None, it is 
                in the test/eval phase. Defaults to None.
            mode (str, optional): This arg indicates that the negative entity 
                will replace the head or tail entity. when it is 'single', it 
                means that entity will not be replaced. Defaults to 'single'.

        Returns:
            head_emb: Head entity embedding.
            relation_emb: Relation embedding.
            tail_emb: Tail entity embedding.
        """
        
        head_emb = self.ent_emb(triples[:, 0])  # [bs, dim]
        relation_emb = self.rel_emb(triples[:, 1])  # [bs, dim]
        tail_emb = self.ent_emb(triples[:, 2])  # [bs, dim]
        
        return self.ent2emb(triples[:, 0]), self.rel2emb(triples[:, 1]), self.ent2emb(triples[:, 2])
    
    def ent2emb(self, entities):
        """Get embedding of entities.

        Args:
            entities (tensor): This tensor save entities id, which dimension is 
                [entities number].

        Returns:
            emb: Entities embedding.
        """
        return self.adapter(self.ent_emb(entities))
    
    def rel2emb(self, relations):
        """Get embedding of relations.

        Args:
            relations (tensor): This tensor save relations id, which dimension is 
                [relations number].

        Returns:
            emb: Relations embedding.
        """
        return self.adapter(self.rel_emb(relations))
    
    def forward(self, triples, mode='none'):

        return 1


if __name__ == '__main__':
    # 将字典转换为ArgumentParser对象
    def dict_to_args(args_dict):
        parser = argparse.ArgumentParser(description="KG Model Arguments")
        for key, value in args_dict.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        return parser.parse_args([])
    
     # 定义参数字典
    args_dict = {
        'margin': 9.0,
        'emb_dim': 500,
        'hidden_dim': 1024,
        'device':'cuda',
        'pretrain_path':'/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/Time-LLM/pretrain_kg/emb.pth',
        'pretrain_emb':True,
        
    }
    args = dict_to_args(args_dict)
    kgmodel=KGModel(args).to(args.device)
    # 可以按照entity索引拿到对应的embedding，然后作为prompt embedding输入到LLM中
    print(kgmodel.ent2emb(torch.tensor([1,2,3,4,5,6,7,8,9,10]).to(args.device)).shape)