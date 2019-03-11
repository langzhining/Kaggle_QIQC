import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, heads=3, seq_len=50, dropout=0.2, is_out=False):
        super(SelfAttention, self).__init__()
        assert hidden_size % heads == 0
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.maxpool = nn.MaxPool2d(kernel_size=(seq_len, 1))
        self.out_layer = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size
        self.heads = heads
        self.attn_size = int(hidden_size / heads)

        self.is_out = is_out

    def transpose_for_scores(self, x, layer):
        x = layer(x)
        new_shape = x.size()[:-1] + (self.heads, self.attn_size)
        x = x.view(*new_shape).permute(0, 2, 1, 3)
        return x

    def forward(self, hidden_stations, attention_mask):
        # (batch_size, seq_len, hidden_size)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        hidden_shape = hidden_stations.size()
        # query: (batch_size, heads, seq_len, attn_size)
        query = self.transpose_for_scores(hidden_stations, self.query)
        key = self.transpose_for_scores(hidden_stations, self.key)
        value = self.transpose_for_scores(hidden_stations, self.value)

        # (batch_size, heads, query_len, key_len)
        attention_weight = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attn_size)
        attention_weight = attention_weight.masked_fill_(attention_mask, -1e9)
        attention_weight = nn.Softmax(dim=-1)(attention_weight)

        attention_weight = self.dropout(attention_weight)

        # (batch_size, heads, query_len, attn_size)
        context = torch.matmul(attention_weight, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*hidden_shape)
        if self.is_out:
            context = context.unsqueeze(1)
            context = self.maxpool(context).squeeze()
            out = self.out_layer(context)
        else:
            out = torch.relu(context)

        return out


class CNN(nn.Module):
    def __init__(self, hidden_size, filters, seq_len, dropout=0.2):
        super(CNN, self).__init__()
        self.layer_1 = nn.Conv2d(1, filters, kernel_size=(1, hidden_size))
        #         self.layer_2 = nn.Conv2d(filters, filters, kernel_size=(seq_len, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(seq_len, 1))
        self.dense = nn.Linear(filters, 1)
        self.dropout = nn.Dropout(dropout)
        self.filters = filters

    def forward(self, x, mask):
        # (batch_size, 1, seq_len, hidden_size)
        x = x.unsqueeze(1)
        cnn_mask = mask.unsqueeze(1).unsqueeze(3)
        x = x.masked_fill_(cnn_mask, 0)
        # (batch_size, filters, seq_len, 1)
        conv1 = F.relu(self.layer_1(x))
        # (batch_size, filters, 1, 1)
        #         out1 = F.relu(self.layer_2(conv1)).squeeze(2).squeeze(2)
        out = self.maxpool(conv1).squeeze()
        #         out = torch.cat([out1, out2], dim=1)
        out = self.dropout(out)
        out = self.dense(out)
        return out


class AttnCNNModel(nn.Module):
    def __init__(self, embedding_matrix, layers, freeze=False):
        super(AttnCNNModel, self).__init__()
        self.embed_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze)
        self.layers = nn.ModuleList(layers)
        # self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            print(name)
            if name.find('embed_layer') > -1:
                print('aaa', name)
                continue
            elif name.find('weight') > -1 and len(param.size()) > 1:
                print('bbb', name)
                nn.init.xavier_uniform_(param)

    def forward(self, x, mask):
        out = self.embed_layer(x)
        for layer in self.layers:
            out = layer(out, mask)
        return out


def model_build(embedding_matrix, config):
    heads = config.heads
    max_seq_len = config.max_seq_len
    cnn_filter = config.cnn_filter
    dropout = config.dropout
    embed_freeze = config.embed_freeze
    embedding_size = len(embedding_matrix[0])
    attn_layer = SelfAttention(hidden_size=embedding_size, heads=heads, seq_len=max_seq_len, dropout=dropout, is_out=False)
    cnn_layer = CNN(embedding_size, cnn_filter, max_seq_len, dropout=dropout)
    model = AttnCNNModel(embedding_matrix, [attn_layer, cnn_layer], freeze=embed_freeze)
    return model