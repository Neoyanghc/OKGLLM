from math import sqrt

import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
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

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # if configs.llm_model == 'LLAMA':
        #     # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
        #     self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        #     self.llama_config.num_hidden_layers = configs.llm_layers
        #     self.llama_config.output_attentions = True
        #     self.llama_config.output_hidden_states = True
        #     try:
        #         self.llm_model = LlamaModel.from_pretrained(
        #             # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        #             'huggyllama/llama-7b',
        #             trust_remote_code=True,
        #             local_files_only=True,
        #             config=self.llama_config,
        #             # load_in_4bit=True
        #         )
        #     except EnvironmentError:  # downloads model from HF is not already done
        #         print("Local model files not found. Attempting to download...")
        #         self.llm_model = LlamaModel.from_pretrained(
        #             # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        #             'huggyllama/llama-7b',
        #             trust_remote_code=True,
        #             local_files_only=False,
        #             config=self.llama_config,
        #             # load_in_4bit=True
        #         )
        #     try:
        #         self.tokenizer = LlamaTokenizer.from_pretrained(
        #             # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
        #             'huggyllama/llama-7b',
        #             trust_remote_code=True,
        #             local_files_only=True
        #         )
        #     except EnvironmentError:  # downloads the tokenizer from HF if not already done
        #         print("Local tokenizer files not found. Atempting to download them..")
        #         self.tokenizer = LlamaTokenizer.from_pretrained(
        #             # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
        #             'huggyllama/llama-7b',
        #             trust_remote_code=True,
        #             local_files_only=False
        #         )
        # elif configs.llm_model == 'GPT2':
        #     self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

        #     self.gpt2_config.num_hidden_layers = configs.llm_layers
        #     self.gpt2_config.output_attentions = True
        #     self.gpt2_config.output_hidden_states = True
        #     try:
        #         self.llm_model = GPT2Model.from_pretrained(
        #             'openai-community/gpt2',
        #             trust_remote_code=True,
        #             local_files_only=True,
        #             config=self.gpt2_config,
        #         )
        #     except EnvironmentError:  # downloads model from HF is not already done
        #         print("Local model files not found. Attempting to download...")
        #         self.llm_model = GPT2Model.from_pretrained(
        #             'openai-community/gpt2',
        #             trust_remote_code=True,
        #             local_files_only=False,
        #             config=self.gpt2_config,
        #         )

        #     try:
        #         self.tokenizer = GPT2Tokenizer.from_pretrained(
        #             'openai-community/gpt2',
        #             trust_remote_code=True,
        #             local_files_only=True
        #         )
        #     except EnvironmentError:  # downloads the tokenizer from HF if not already done
        #         print("Local tokenizer files not found. Atempting to download them..")
        #         self.tokenizer = GPT2Tokenizer.from_pretrained(
        #             'openai-community/gpt2',
        #             trust_remote_code=True,
        #             local_files_only=False
        #         )
        # elif configs.llm_model == 'BERT':
        #     self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

        #     self.bert_config.num_hidden_layers = configs.llm_layers
        #     self.bert_config.output_attentions = True
        #     self.bert_config.output_hidden_states = True
        #     try:
        #         self.llm_model = BertModel.from_pretrained(
        #             'google-bert/bert-base-uncased',
        #             trust_remote_code=True,
        #             local_files_only=True,
        #             config=self.bert_config,
        #         )
        #     except EnvironmentError:  # downloads model from HF is not already done
        #         print("Local model files not found. Attempting to download...")
        #         self.llm_model = BertModel.from_pretrained(
        #             'google-bert/bert-base-uncased',
        #             trust_remote_code=True,
        #             local_files_only=False,
        #             config=self.bert_config,
        #         )

        #     try:
        #         self.tokenizer = BertTokenizer.from_pretrained(
        #             'google-bert/bert-base-uncased',
        #             trust_remote_code=True,
        #             local_files_only=True
        #         )
        #     except EnvironmentError:  # downloads the tokenizer from HF if not already done
        #         print("Local tokenizer files not found. Atempting to download them..")
        #         self.tokenizer = BertTokenizer.from_pretrained(
        #             'google-bert/bert-base-uncased',
        #             trust_remote_code=True,
        #             local_files_only=False
        #         )
        # else:
        #     raise Exception('LLM model is not defined')
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'Deepseek':
            self.deepseek_config = AutoConfig.from_pretrained('/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/deepseek7b')

            self.deepseek_config.num_hidden_layers = configs.llm_layers
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            try:
                self.llm_model = AutoModel.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.deepseek_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.deepseek_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/deepseek7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'OceanGPT':
            self.oceangpt_config = AutoConfig.from_pretrained('/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/oceangpt')

            self.oceangpt_config.num_hidden_layers = configs.llm_layers
            self.oceangpt_config.output_attentions = True
            self.oceangpt_config.output_hidden_states = True
            # try:
            #     self.llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #         '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/oceangpt',
            #         trust_remote_code=True,
            #         local_files_only=True,
            #         config=self.oceangpt_config,
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            
            print("Local model files not found. Attempting to download...")
            self.llm_model = AutoModel.from_pretrained(
                '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/oceangpt',
                trust_remote_code=True,
                local_files_only=False,
                config=self.oceangpt_config,
            )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/oceangpt',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/oceangpt',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    '/remote-home/share/dmb_nas2/yhc/2025/LFM/openai-community/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
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

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.ts_model = DLinear(configs)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc_original, x_mark_enc, x_dec, x_mark_dec):

        x_enc_norm = self.normalize_layers(x_enc_original, 'norm')

        B, T, N = x_enc_norm.size()
        x_enc = x_enc_norm.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        # ocean_knowledge = np.loadtxt('/remote-home/share/dmb_nas2/yhc/2025/OceanKG/LLM_KG_Prediction/KG-LLM/grid_centers.txt', delimiter=',', dtype=str)
        # # 针对每一个序列进行描述
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

            # ocean_type = ocean_knowledge[b].tolist()[-2]
            # ocean_zone = ocean_knowledge[b].tolist()[-1]
            # ocean_prompt_ = (
            #     f"<|start_prompt|>Dataset description: {self.description} of {ocean_type}, at {ocean_zone},"
            #     f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
            # )

            # prompt.append(ocean_prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # origianl verison
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # # revised version yhc, only text for LLM, TS concat at back 
        # # x_enc = x_enc.permute(0, 2, 1).contiguous()
        # # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # # enc_out = self.ts_model(x_enc_original, x_mark_enc, x_dec, x_mark_dec)
        # # # print('enc_out:', enc_out.shape)
        # # llm_dec_out = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        # # # print('llm_dec_out:', llm_dec_out.shape)

        # # # 将 enc_out 的形状调整为与 llm_dec_out 兼容
        # # enc_out_expanded = enc_out.expand(-1, -1, llm_dec_out.size(2))  # [256, 32, 768]

        # # # 在特征维度拼接 (dim=1)
        # # # print('enc_out_expanded:', enc_out_expanded.shape)
        # # concat_output = torch.cat([enc_out_expanded, llm_dec_out], dim=1) # [256, 102, 768]
        # # # print('concat_output:', concat_output.shape)
        # # dec_out = concat_output[:, :, :self.d_ff]


        # # revised version yhc, only text for LLM, TS concat LLM at back 
        # # llm_dec_out = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        # # print('llm_dec_out:', llm_dec_out.shape)
        # # print( 'x_enc:', x_enc_norm.shape)
        # # x_enc_expanded= x_enc_norm.expand(-1, -1, llm_dec_out.size(2))
        # # x_enc_ts = torch.cat([llm_dec_out, x_enc_expanded], dim=1) # [256, 102, 768]
        # # print('x_enc_ts:', x_enc_ts.shape)
        # # enc_out = self.ts_model(x_enc_ts, x_mark_enc, x_dec, x_mark_dec)
        # # print('enc_out:', enc_out.shape)
        # # print('llm_dec_out:', llm_dec_out.shape)
        # # 将 enc_out 的形状调整为与 llm_dec_out 兼容
        # # print('concat_output:', concat_output.shape)
        # # dec_out = concat_output[:, :, :self.d_ff]

        # revised version yhc, TS embeeding and text for LLM
        # x_enc = x_enc.permute(0, 2, 1).contiguous()
        # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out = self.ts_model(x_enc_original, x_mark_enc, x_dec, x_mark_dec)
        # print('enc_out:', enc_out.shape)
        # print('prompt_embeddings:', prompt_embeddings.shape)
        # enc_out_expanded = enc_out.expand(-1, -1, prompt_embeddings.size(2))  # [256, 32, 768]
        # # print('enc_out_expanded:', enc_out_expanded.shape)
        # llama_enc_out = torch.cat([prompt_embeddings, enc_out_expanded], dim=1)
        # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

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
