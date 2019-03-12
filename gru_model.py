import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_cnn_model import SelfAttention


class GlobalMaxPooling1D(nn.Module):
    def __init__(self, pool_dim=1):
        super(GlobalMaxPooling1D, self).__init__()
        self.pool_dim = pool_dim

    def forward(self, inputs):
        z, _ = torch.max(inputs, self.pool_dim)
        return z

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hiddens, mask):
        # hiddens: (batch, seq_len, hidden_size)
        # weight : (batch, seq_len, 1)
        # out : (batch, hidden_size)
        weight = self.query(hiddens)
        weight = F.softmax(weight, dim=1)
        out = (hiddens * weight).sum(dim=1)
        return out

class CNN(nn.Module):
    def __init__(self, hidden_size, filters=64, dropout=0.2):
        super(CNN, self).__init__()
        self.layer_1 = nn.Conv2d(1, filters, kernel_size=(1, hidden_size))
        self.maxpool = GlobalMaxPooling1D(2)
        #         self.layer_2 = nn.Conv2d(filters, filters, kernel_size=(seq_len, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=(seq_len, 1))
        # self.dense = nn.Linear(filters, 1)
        # self.dropout = nn.Dropout(dropout)
        self.filters = filters

    def forward(self, x, *args):
        # (batch_size, 1, seq_len, hidden_size)
        x = x.unsqueeze(1)

        # cnn_mask = mask.unsqueeze(1).unsqueeze(3)
        # x = x.masked_fill_(cnn_mask, 0)
        # (batch_size, filters, seq_len, 1)
        conv1 = torch.relu(self.layer_1(x))
        # (batch_size, filters, 1, 1)
        #         out1 = F.relu(self.layer_2(conv1)).squeeze(2).squeeze(2)
        out = self.maxpool(conv1).squeeze()
        #         out = torch.cat([out1, out2], dim=1)
        # out = self.dropout(out)
        # out = self.dense(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, pretrained_embedding, proj_dim, rnn_dim, n_layers, bidirectional, dense_dim,
                 padding_idx=0, fix_embedding=True, attn_layer=None, cnn_layer=None, dropout=0.4,
                 n_out=1):
        super(GRUModel, self).__init__()
        # self.n_vocab = n_vocab
        self.attn_layer = attn_layer
        self.cnn_layer = cnn_layer
        self.embed_dim = len(pretrained_embedding[0])
        self.n_layers = n_layers
        self.dense_dim = dense_dim
        self.n_out = n_out
        self.bidirectional = bidirectional
        self.fix_embedding = fix_embedding
        self.padding_idx = padding_idx
        # if pretrained_embedding is not None:
        self.embed = nn.Embedding.from_pretrained(pretrained_embedding, freeze=fix_embedding)
        self.embed.padding_idx = self.padding_idx
        # else:
        #     self.embed = nn.Embedding(self.n_vocab, self.embed_dim, padding_idx=self.padding_idx)
        self.proj = nn.Linear(self.embed_dim, proj_dim)
        self.proj_act = nn.ReLU()
        self.gru = nn.GRU(proj_dim, rnn_dim, self.n_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.pooling = GlobalMaxPooling1D()
        in_dim = 2 * rnn_dim if self.bidirectional else rnn_dim
        self.dense = nn.Linear(in_dim, dense_dim)
        self.dense_act = nn.ReLU()
        self.out_linear = nn.Linear(dense_dim, n_out)
        self.dropout = nn.Dropout(dropout)
        self.embed_dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.find('embed') > -1:
                continue
            elif name.find('weight') > -1 and len(param.size()) > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs, *args):
        # inputs: (bs, max_len)
        x = self.embed(inputs)
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.embed_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        x = self.proj_act(self.proj(x))
        x, hidden = self.gru(x)
        if self.attn_layer is not None:
            x = self.attn_layer(x, args[0])
        elif self.cnn_layer is not None:
            x = self.cnn_layer(x)
        else:
            x = self.pooling(x)
        x = self.dense_act(self.dense(x))
        x = self.dropout(x)
        x = self.out_linear(x)
        return x


def model_build(embedding_matrix, config):
    proj_dim = config.proj_dim
    rnn_dim = config.rnn_dim
    n_layers = config.n_layers
    bidirectional = config.bidirectional
    dense_dim = config.dense_dim
    fix_embedding = config.fix_embedding
    hidden_size = rnn_dim * 2 if bidirectional else rnn_dim
    attn_layer = Attention(hidden_size)
    # attn_layer = SelfAttention(attn_size, heads=4)
    # cnn_layer = CNN(hidden_size, hidden_size)
    cnn_layer = None
    model = GRUModel(embedding_matrix, proj_dim, rnn_dim, n_layers, bidirectional, dense_dim,
                     fix_embedding=fix_embedding, padding_idx=0, attn_layer=attn_layer)
    return model