from math import sqrt

import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from models.DLinear import Model as DLinear

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, kgmodel=None,kgdata=None,patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.kgmodel = kgmodel
        self.kgdata = kgdata
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('/openai-community/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    '/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    '/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    '/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    '/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'Deepseek':
            self.deepseek_config = AutoConfig.from_pretrained('/openai-community/deepseek7b')

            self.deepseek_config.num_hidden_layers = configs.llm_layers
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            try:
                self.llm_model = AutoModel.from_pretrained(
                    '/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.deepseek_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    '/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.deepseek_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('/openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    '/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    '/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    '/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    '/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('/openai-community/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    '/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    '/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    '/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    '/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'OceanGPT':
            self.oceangpt_config = AutoConfig.from_pretrained('/openai-community/oceangpt')

            self.oceangpt_config.num_hidden_layers = configs.llm_layers
            self.oceangpt_config.output_attentions = True
            self.oceangpt_config.output_hidden_states = True
            # try:
            #     self.llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #         '/openai-community/oceangpt',
            #         trust_remote_code=True,
            #         local_files_only=True,
            #         config=self.oceangpt_config,
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            
            print("Local model files not found. Attempting to download...")
            self.llm_model = AutoModel.from_pretrained(
                '/openai-community/oceangpt',
                trust_remote_code=True,
                local_files_only=False,
                config=self.oceangpt_config,
            )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/openai-community/oceangpt',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/openai-community/oceangpt',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            # pass
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # freeze the LLM model, yhc
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'sea surface temperature with one week sample rate.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        # self.dff_layer= nn.Linear(self.d_llm, self.d_ff)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        # self.kg_cross_atten = CrossAttention(self.d_llm, 1)
        
        self.ts_model = DLinear(configs)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_descriptions, kg_idx,mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_outs = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_descriptions,kg_idx)
            dec_out, tse = dec_outs[0], dec_outs[1]
            return dec_out[:, -self.pred_len:, :], tse
        return None

    def forecast(self, x_enc_original, x_mark_enc, x_dec, x_mark_dec, batch_descriptions,kg_idx):

        x_enc_norm = self.normalize_layers(x_enc_original, 'norm')

        B, T, N = x_enc_norm.size()
        # print(x_enc_norm.shape)
        x_enc = x_enc_norm.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # print(x_enc.shape)
        # min_values = torch.min(x_enc, dim=1)[0]
        # max_values = torch.max(x_enc, dim=1)[0]
        # medians = torch.median(x_enc, dim=1).values
        # lags = self.calcute_lags(x_enc)
        # trends = x_enc.diff(dim=1).sum(dim=1)
        
       
        # prompt = []
        # kg_embeddings = torch.ones(x_enc.shape[0],1,self.d_llm).to(x_enc.device)
        # # 针对每一个序列进行描述
        # for b in range(x_enc.shape[0]):
        #     # prompt_ = (
        #     #     f"<|start_prompt|>Dataset description: {self.description}"
        #     #     f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
        #     #     "Input statistics: "
        #     #     f"min value {min_values_str}, "
        #     #     f"max value {max_values_str}, "
        #     #     f"median value {median_values_str}, "
        #     #     f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
        #     #     f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
        #     # )
        #     # prompt.append(prompt_)
        #     ocean_idx=batch_descriptions[b].tolist()[0]
        #     # 1hop的邻居不加速
        #     # l_hop_neighbors=self.kgdata.get_l_hop_triples_dgl(self.kgdata.dgl_graph,
        #     #                                                   self.kgdata.ent2id['region_'+ocean_idx],1)
        #     # 1hop的邻居加速
        #     l_hop_neighbors=self.kgdata.region_neighbors['region_'+ocean_idx]
                                                             
        #     l_hop_neighbors_text=self.kgdata.triple2text(l_hop_neighbors).replace('_',' ')
        #     # ocean_type = batch_descriptions[b].tolist()[-2]
        #     # ocean_zone = batch_descriptions[b].tolist()[-1]
        #     ocean_prompt_ = (
        #         f"<|start_prompt|> Description: {l_hop_neighbors_text} <|end_prompt|>"
        #     )
            
        #     prompt.append(ocean_prompt_)
           
        #     kg_embeddings[b][0] = self.kgmodel.ent2emb(torch.tensor(self.kgdata.ent2id['region_'+ocean_idx]).to(x_enc.device))
        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        #############################################
        kg_embeddings = self.kgmodel.ent2emb(kg_idx).unsqueeze(1)
        prompt=batch_descriptions
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids
        # import ipdb;ipdb.set_trace()
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # origianl verison
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        ts_embeeding, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        
        ts_embeeding = self.reprogramming_layer(ts_embeeding, source_embeddings, source_embeddings)
#        # kg_embeddings = self.kg_cross_atten(kg_embeddings, source_embeddings)
        #############################################
        llama_enc_out = torch.cat([prompt_embeddings,kg_embeddings,ts_embeeding], dim=1)
        # llama_enc_out = torch.cat([kg_embeddings,prompt_embeddings, enc_out], dim=1)
        # llama_enc_out = torch.cat([kg_embeddings,enc_out,prompt_embeddings], dim=1)
        # llama_enc_out = ts_embeeding
        #############################################
        # tse = [kg_embeeding,ts_embeeding]
        tse = [ts_embeeding,ts_embeeding]
    
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state

        # 截取 dec_out 的前 self.d_ff 个维度
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = self.dff_layer(dec_out)

        # 将 dec_out 调整成目标形状
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # **增加残差连接**
        # 假设我们使用 enc_out 作为残差分支的输入
        # 首先调整 enc_out 的形状以匹配 dec_out 的形状
        residual_enc_out = ts_embeeding.unsqueeze(-1)  # 添加一个维度
        residual_enc_out = torch.nn.functional.interpolate(
            residual_enc_out, size=(dec_out.shape[-2], dec_out.shape[-1]), mode='bilinear', align_corners=False
        )  # 插值以匹配 dec_out 的最后两维

        # 添加残差连接
        dec_out = dec_out + residual_enc_out

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return [dec_out, tse]

    def calcute_lags(self, x_enc):
        # top k frenquency is used in the Time-LLM
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # 先训练这个projection      

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding



class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query_batch, context):
        
        BS = query_batch.size(0)
        batch_context = context.unsqueeze(0).expand(BS, -1, -1)


        out, _ = self.attn(query_batch, batch_context, batch_context)

        return out
