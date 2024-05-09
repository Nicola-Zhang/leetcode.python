#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
读取一个文件input_file.txt
已知文件里存储了已经分词后的中文语料，类似

------------
中华 人民 共和国 成立 于 1 9 4 9 年...
北京 是 中华 人民 共和国 的 首都。新 中国 成立 以后...
...
...
...
------------

请统计语料中的所有不同的ngram频次（n为可指定参数），按倒序排列并打印输出到一个文件output_file.txt
例如，当n=2时，上面的例子中包含的2-gram有：
'中华 人民', '人民 共和国', '共和国 成立', '成立 于', ...
"""

from collections import Counter
from typing import Any, AnyStr, Tuple, List, Dict, Optional, Union


def load_txt(file_path):
    with open(file_path, "r", encoding='utf-8') as fr:
        content = fr.read()
        return content.strip()


def token_n_gram(dataset, n):
    word_list = []
    for i in range(len(dataset)-n):
        word_list.append(''.join(dataset[i+n]))

    words_count = Counter(dataset)
    words_n_count = Counter(word_list)

    words_list_len = len(word_list)
    unique_words_len = len(words_count)
    unique_bigram_len = len(words_n_count)


    def prob_1(word):
        return (words_count[word] + 1) / (words_list_len + unique_words_len)

    def prob_2(word1, word2):
        return (words_n_count[word1 + word2] + 1) / (words_count[word1] + unique_bigram_len)

    probility = prob_1(dataset[0])
    for index in range(len(dataset)):
        probility *= prob_2(dataset[index], dataset[index+1])




