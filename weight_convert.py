# -*- coding: utf-8 -*-

import tensorflow as tf
from gpt2_torch import GPT2
import numpy as np
import re
import torch
from pathlib import Path

def init_torch_model_weight_from_tf_ckpt(pt_model, tf_ckpt_path, checkpoint_dir):
    tf_ckpt_path = tf.train.latest_checkpoint(tf_ckpt_path)

    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())

    weight_map = tf_param_name_to_torch_map()
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        if "model/h" in name:
            abstract_key = re.sub(r'h(\d+)', "{}", name)
            layer_num = re.search(r'\d+', name).group(0)
            new_key = weight_map.get(abstract_key)
            if new_key is None:
                continue
            else:
                new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[name]
        
        if str(name).endswith("/w"): # need transpose
            array = array.transpose(1, 0) 
        
        value = torch.from_numpy(array)
        if isinstance(new_key, list):
            for key in new_key:
                # print(f"{name} -> {key}")
                new_pt_params_dict[key] = value
                assert key in current_pt_params_dict.keys() , f"key {key} not found in torch model."
                assert current_pt_params_dict[key].shape == value.shape, f"{key} shape not match! current shape[{value.shape}], torch model shape[{current_pt_params_dict[key].shape}]"
        else:
            print(f"{name} -> {new_key}")
            new_pt_params_dict[new_key] = value
            assert new_key in current_pt_params_dict.keys() , f"key {new_key} not found in torch model."
            assert current_pt_params_dict[new_key].shape == value.shape, f"{new_key} shape not match! current shape[{value.shape}], torch model shape[{current_pt_params_dict[new_key].shape}]"
    
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pt'}")
    torch.save(new_pt_params_dict, checkpoint_dir / "model.pt")


def tf_param_name_to_torch_map():
    weight_map = {
        "model/wte": ["token_embedding.weight", "linear.weight"],
        "model/wpe": "position_embedding.weight",
        "model/ln_f/b": "norm.beta",
        "model/ln_f/g": "norm.gamma",
        "model/{}/attn/c_attn/b": "transformer_block.{}.mulit_head_attention.c_attn.bias",
        "model/{}/attn/c_attn/w": "transformer_block.{}.mulit_head_attention.c_attn.weight",
        "model/{}/attn/c_proj/b": "transformer_block.{}.mulit_head_attention.c_proj.bias",
        "model/{}/attn/c_proj/w":"transformer_block.{}.mulit_head_attention.c_proj.weight",
        "model/{}/ln_1/b": "transformer_block.{}.norm1.beta",
        "model/{}/ln_1/g": "transformer_block.{}.norm1.gamma",
        "model/{}/ln_2/b": "transformer_block.{}.norm2.beta",
        "model/{}/ln_2/g": "transformer_block.{}.norm2.gamma",
        "model/{}/mlp/c_fc/w": "transformer_block.{}.ffn.c_fc.weight",
        "model/{}/mlp/c_fc/b": "transformer_block.{}.ffn.c_fc.bias",
        "model/{}/mlp/c_proj/w": "transformer_block.{}.ffn.c_proj.weight",
        "model/{}/mlp/c_proj/b": "transformer_block.{}.ffn.c_proj.bias"
    }
    return weight_map




if __name__ == "__main__":
    tf_ckpt_path = "/models/gpt2-tf/124M"
    config = {
        "n_vocab": 50257,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12
    }
    torch_model = GPT2(config=config)
    init_torch_model_weight_from_tf_ckpt(torch_model, tf_ckpt_path, Path("."))