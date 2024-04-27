#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
=================================================
@Project -> File ：my_leetcode.python -> test.py.py
@Author ：Nicola.Zhang
@E-mail: xuzhang0423@gmail.com
@Date ：2024/4/25 19:37 
@Desc ：
==================================================
'''
from typing import Any, AnyStr, Tuple, List, Dict, Optional, Union

def dataloader(inputs):
    voc_table = voc2id(inputs)

    output = []

    bsz = 2
    data_len = len(inputs)
    for s_idx in range(0, data_len, bsz):
        e_idx = s_idx+2 if s_idx+2 < data_len else data_len
        batch = tokenizer(voc_table, inputs[s_idx: e_idx], max_length=5, batch_size=bsz)
        output.append(batch)

    return output


def voc2id(inputs):
    id_dict = {}
    for input in inputs:
        for i in input:
            if i not in id_dict:
                id_dict[i] = len(id_dict)+1
    return id_dict


def tokenizer(id_dict, inputs, max_length, batch_size):
    batch_inputs = []
    for input in inputs:
        input_id = [id_dict[i] for i in input]
        if len(input_id) >= max_length:
            input_id = input_id[:max_length]
        else:
            for _ in range(max_length-len(input_id)):
                input_id.append(0)
        batch_inputs.append(input_id)

    if len(batch_inputs) < batch_size:
        for _ in range(batch_size-len(batch_inputs)):
            batch_inputs.append([0]*max_length)
    return batch_inputs

if __name__ == '__main__':
    case = ["dsdd", "dsssddd", "dddweereds", "dds", "eewierew"]
    output = dataloader(case)
    print(output)
    print(len(output), len(output[0]), len(output[0][0]))