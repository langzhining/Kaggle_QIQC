import pandas as pd
import numpy as np
import re
import os
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset

data_path = 'input/'
word2vec_path = data_path+'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'


# 清理数据
def clean_punct(s):
    puncts = ',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'
    for punct in puncts:
        s = s.replace(punct, ' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def hash_number(s):
    s = re.sub('[0-9]{5,}', '#####', s)
    s = re.sub('[0-9]{4}', '####', s)
    s = re.sub('[0-9]{3}', '###', s)
    s = re.sub('[0-9]{2}', '##', s)
    return s

def clean_text(texts):
    new_texts = []
    for s in texts:
        s = clean_punct(s)
        s = hash_number(s)
        s = s.lower()
        new_texts.append(s)
    return new_texts


# 从已有词向量表中找到对应的单词，并生成子词典（tokenizer）
def get_vocab_by_embed(full_tokenizer, embed_dict):
    word_list = []
    for word in full_tokenizer.word_counts.keys():
        if word in embed_dict:
            word_list.append(word)
    words = ' '.join(word_list)
    sub_vocab = Tokenizer(lower=False)
    sub_vocab.fit_on_texts([words])
    return sub_vocab

# 生成与单词索引匹配的词向量
def get_embedding_matrix(tokenizer, embed_dict):
    vector_size = len(embed_dict['known'])
    embedding_shape = (len(tokenizer.word_index)+1, vector_size)
    embedding_matrix = np.zeros(embedding_shape)
    indexes = []
    for word, index in tokenizer.word_index.items():
        embedding_matrix[index] = embed_dict[word]
        indexes.append(index)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    return embedding_matrix


# 将文本转化为tensor
def texts_to_tensor(texts, tokenizer, maxlen=50):
    seqs = tokenizer.texts_to_sequences(texts)
    seqs_padded = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='pre', value=0)
    seqs_padded = torch.tensor(seqs_padded, dtype=torch.int64)
    mask = seqs_padded == 0
    mask = mask
    return seqs_padded, mask


# 构建dataloader，用于torch训练和测试用
def get_dataloader(x, mask, y=None,training=True, batch_size=32,
                   weights=None, num_samples=None, drop_last=False):
    if y is None:
        data = TensorDataset(x, mask)
    else:
        data = TensorDataset(x, mask,y)
    if training:
        if weights is None:
            sampler = RandomSampler(data)
        else:
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, shuffle=False, batch_size=batch_size, drop_last=drop_last)
    return dataloader


def get_train_data():
    train_df = pd.read_csv(data_path + 'train.csv')
    train_texts = list(train_df.question_text.values)
    train_texts = clean_text(train_texts)

    word2vec_dict = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    full_tokenizer = Tokenizer(lower=False)
    full_tokenizer.fit_on_texts(train_texts)

    word2vec_tokenizer = get_vocab_by_embed(full_tokenizer, word2vec_dict)

    word2vec_matrix = get_embedding_matrix(word2vec_tokenizer, word2vec_dict)

    train_x, train_mask = texts_to_tensor(train_texts, word2vec_tokenizer)
    train_y = torch.tensor(train_df.target.values, dtype=torch.float32)

    return (train_x, train_mask, train_y), [(word2vec_matrix, word2vec_tokenizer),]


def split_train_eval(train_x, train_mask, train_y):
    eval_x = train_x[1000000:]
    eval_mask = train_mask[1000000:]
    eval_y = train_y[1000000:]

    train_x = train_x[:1000000]
    train_mask = train_mask[:1000000]
    train_y = train_y[:1000000]
    return (train_x, train_mask, train_y), (eval_x, eval_mask, eval_y)
