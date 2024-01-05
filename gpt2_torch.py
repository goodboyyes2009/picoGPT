# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class GPT2(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['n_vocab'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['n_ctx'],  config['n_embd'])
        self.transformer_block = nn.ModuleList([TransformerBlock(config['n_head'], config['n_embd']) for _ in range(config['n_layer'])])
        self.norm = LayerNorm(n_embd=config['n_embd'])
        self.linear = nn.Linear(config['n_embd'], config['n_vocab'],bias=False)

    
    def forward(self, inputs):
        # token + positional embeddings
        token_embd = self.token_embedding(inputs)
        position_embd = self.position_embedding(torch.arange(0, len(inputs), dtype=torch.int64))
        x = token_embd + position_embd
        for block in self.transformer_block:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, n_embd) -> None:
        super(TransformerBlock, self).__init__()
        self.mulit_head_attention = MultiHeadAttention(num_heads, n_embd)
        self.ffn = FFN(n_embd=n_embd)
        self.norm1 = LayerNorm(n_embd=n_embd)
        self.norm2 = LayerNorm(n_embd=n_embd)
    

    def forward(self, x):
        norm = self.norm1(x)
        x = x + self.mulit_head_attention(norm)
        norm = self.norm2(x)
        x = x + self.ffn(norm)
        return x

class FFN(nn.Module):
    def __init__(self, n_embd) -> None:
        super(FFN, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_embd * 4)
        self.c_proj = nn.Linear(n_embd * 4, n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        return self.c_proj(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x: torch.Tensor):
        seq_len = x.shape[0] # shape=[seq_len, n_embd]
        x = self.c_attn(x) # shape=[seq_len, 3 * n_embd]
        # split qkv into key, query, value
        # qkv = x.view(seq_len, 3, -1).transpose(0, 1) # [seq_len, 3 * n_embd] -> [3, seq_len, n_embd]
        qkv = rearrange(x, 'seq_len (n n_embd) -> n seq_len n_embd', n=3)
        # qkv_heads = qkv.view(3, seq_len, self.num_heads, -1).transpose(1,2) # [3, num_heads, seq_len, n_embd/num_heads]
        qkv_heads = rearrange(qkv, 'n seq_len (n_heads head_embd) -> n n_heads seq_len head_embd', n_heads=self.num_heads)

        # causal mask to hide future inputs from being attended to

        causal_mask = (1 - torch.tril(torch.ones(x.shape[0], x.shape[0]).type_as(x))) * -1e10  # [n_seq, n_seq]

        attention_out = self.attention(qkv_heads[0], qkv_heads[1], qkv_heads[2], causal_mask) # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
        # merge heads
        concat_all_attentions = torch.hstack([attention_out[i] for i in range(self.num_heads)]) # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
        x = self.c_proj(concat_all_attentions) # [n_seq, n_embd] -> [n_seq, n_embd]
        return x
        

    def attention(self, q:torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        # qk = torch.bmm(q, k.transpose(1, 2)) # [n_head, seq_len, n_embd/n_head] * [n_head, n_embd/n_head, seq_len] = [n_head, seq_len, seq_len]
        qk = einsum(q, k, 'n_head seq_len head_embd, n_head seq_len_1 head_embd -> n_head seq_len seq_len_1')
        qk = qk / torch.sqrt(torch.tensor(q.shape[-1]))
        # qk shape=[n_head, seq_len, seq_len]
        # mask shape=[seq_len, seq_len]
        qk = qk + mask
        # qk_clone = F.softmax(qk) # dim=0
        qk = F.softmax(qk, dim=-1)
        attention = einsum(qk, v, 'n_head seq_len seq_len_1, n_head seq_len_1 head_embd -> n_head seq_len head_embd')
        return attention
        


class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x-mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x-mean) / torch.sqrt(variance + self.eps)
        return self.gamma * x + self.beta


def gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2) / torch.pi) * (x + 0.044715 * x**3)))
    


def generate(model, inputs, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = model(inputs)  # model forward pass

        next_id = torch.argmax(logits[-1])  # greedy sampling

        inputs = torch.cat((inputs, next_id.unsqueeze(0))) # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids



def main(prompt: str, n_tokens_to_generate, model_size: str, models_dir: str):
    from utils import load_encoder_and_hparams

    # load encoder and hparams from the released open-ai gpt-2 files
    encoder, hparams = load_encoder_and_hparams(model_size, models_dir)

    # load model from disk
    gpt2 = GPT2(hparams)
    params_dict = torch.load("./model.pt")
    
    gpt2.load_state_dict(params_dict)
    gpt2.eval()

    # # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # wrapper tensor
    input_ids = torch.from_numpy(np.array(input_ids, dtype=np.int64)).to(torch.int64)
    
    # generate output ids
    output_ids = generate(gpt2, input_ids, n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids.detach().cpu().numpy())
    
    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)