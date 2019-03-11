import time
import os
import pandas as pd
import numpy as np
import random
import copy

import re
from gensim.models import KeyedVectors
import gc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from sklearn.metrics import f1_score
from preprocess import get_vocab_by_embed, get_embedding_matrix


t0 = time.time()

data_path = 'input/'
seed = 2077

# Preprocessing
maxlen = 50
lower = False
trunc = 'pre'
max_features = 130000
n_vocab = max_features
clean_num = 0

# Training
n_models = 6
epochs = 8
batch_size = 512
drop_last = True

hidden_dim = 128

# Embedding
fix_embedding = True
unk_uni = True  # Initializer for unknown words
n_embed = 2
embed_dim = n_embed * 300
proj_dim = hidden_dim

# GRU
bidirectional = True
n_layers = 1
rnn_dim = hidden_dim

# The second last Linear layer
dense_dim = 2 * rnn_dim if bidirectional else rnn_dim

# EMA
mu = 0.9
updates_per_epoch = 10

# Test set
threshold = 0.37
test_batch_size = 8 * batch_size


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def get_param_size(model, trainable=True):
    if trainable:
        psize = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    else:
        psize = np.sum([np.prod(p.size()) for p in model.parameters()])
    return psize


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856
class EMA():
    def __init__(self, model, mu, level='batch', n=1):
        """
        level: 'batch' or 'epoch'
          'batch': Update params every n batches.
          'epoch': Update params every epoch.
        """
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level is 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n

    def on_epoch_end(self, model):
        if self.level is 'epoch':
            self._update(model)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, inputs):
        z, _ = torch.max(inputs, 1)
        return z

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GRUModel(nn.Module):
    def __init__(self, n_vocab, embed_dim, proj_dim, rnn_dim, n_layers, bidirectional, dense_dim,
                 padding_idx=0, pretrained_embedding=None, fix_embedding=True,
                 n_out=1):
        super(GRUModel, self).__init__()
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dense_dim = dense_dim
        self.n_out = n_out
        self.bidirectional = bidirectional
        self.fix_embedding = fix_embedding
        self.padding_idx = padding_idx
        if pretrained_embedding is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embedding, freeze=fix_embedding)
            self.embed.padding_idx = self.padding_idx
        else:
            self.embed = nn.Embedding(self.n_vocab, self.embed_dim, padding_idx=self.padding_idx)
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.proj_act = nn.ReLU()
        self.gru = nn.GRU(proj_dim, rnn_dim, self.n_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.pooling = GlobalMaxPooling1D()
        in_dim = 2 * rnn_dim if self.bidirectional else rnn_dim
        self.dense = nn.Linear(in_dim, dense_dim)
        self.dense_act = nn.ReLU()
        self.out_linear = nn.Linear(dense_dim, n_out)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.find('embed') > -1:
                continue
            elif name.find('weight') > -1 and len(param.size()) > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs):
        # inputs: (bs, max_len)
        x = self.embed(inputs)
        x = self.proj_act(self.proj(x))
        x, hidden = self.gru(x)
        x = self.pooling(x)
        x = self.dense_act(self.dense(x))
        x = self.out_linear(x)
        return x

    def predict(self, dataloader):
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                X_batch, = batch
                preds.append(self.forward(X_batch).data.cpu())
        return torch.cat(preds)

    def predict_proba(self, dataloader):
        return torch.sigmoid(self.predict(dataloader)).data.numpy()


def get_dataloader(x, y=None, weights=None, num_samples=None, batch_size=32,
                   dtype_x=torch.float, dtype_y=torch.float, training=True,
                   drop_last=False):
    x_tensor = torch.tensor([x_1 for x_1 in x], dtype=dtype_x)
    if y is None:
        data = TensorDataset(x_tensor)
    else:
        y_tensor = None if y is None else torch.tensor([y_1 for y_1 in y], dtype=dtype_y)
        data = TensorDataset(x_tensor, y_tensor)
    if training:
        if weights is None:
            sampler = RandomSampler(data)
        else:
            sampler = WeightedRandomSampler(weights, num_samples)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, shuffle=False, batch_size=batch_size,
                            drop_last=drop_last)
    return dataloader


def run_epoch(model, dataloader, optimizer, callbacks=None,
              criterion=nn.BCEWithLogitsLoss(), verbose_step=10000):
    t1 = time.time()
    tr_loss = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        x_batch, y_batch = batch
        model.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs[:, 0], y_batch.float())
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        if callbacks is not None:
            for func in callbacks:
                func.on_batch_end(model)
        if (step + 1) % verbose_step == 0:
            loss_now = tr_loss / (step + 1)
            print(f'step:{step+1} loss:{loss_now:.7f} time:{time.time() - t1:.1f}s')
    if callbacks is not None:
        for func in callbacks:
            func.on_epoch_end(model)
    return tr_loss / (step + 1)


def load_glove(word_index, max_features, unk_uni):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = data_path + 'embeddings/glove.840B.300d/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))

    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns glove', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


def load_wiki(word_index, max_features, unk_uni):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = data_path + 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE,encoding='utf-8') if len(o) > 100)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))

    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns wiki', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


def load_parag(word_index, max_features, unk_uni):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = data_path + 'embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    embeddings_index = dict(get_coefs(*o.split(' '))
                            for o in open(EMBEDDING_FILE, encoding='utf8', errors='ignore')
                            if len(o) > 100)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))
    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns parag', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def load_ggle(word_index, max_features, unk_uni):
    EMBEDDING_FILE = data_path + 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    embed_size = embeddings_index.get_vector('known').size

    unknown_words = []
    nb_words = min(max_features, len(word_index))
    if unk_uni:
        embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix[i] = embedding_vector
        else:
            word_lower = word.lower()
            if word_lower in embeddings_index:
                embedding_matrix[i] = embeddings_index.get_vector(word_lower)
            else:
                unknown_words.append((word, i))

    print('\nTotal unknowns ggle', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


def load_all_embeddings(tokenizer, max_features, clean_num=False, unk_uni=True):
    word_index = tokenizer.word_index
    if clean_num == 2:
        ggle_word_index = {}
        for word, i in word_index.items():
            ggle_word_index[clean_numbers(word)] = i
    else:
        ggle_word_index = word_index

    # embedding_matrix_1, u1 = load_glove(word_index, max_features, unk_uni)
    # embedding_matrix_2, u2 = load_wiki(word_index, max_features, unk_uni)
    # embedding_matrix_3, u3 = load_parag(word_index, max_features, unk_uni)
    # embedding_matrix_4, u4 = load_ggle(ggle_word_index, max_features, unk_uni)
    # embedding_matrix = np.concatenate((embedding_matrix_1,
    #                                    embedding_matrix_2,
    #                                    embedding_matrix_3,
    #                                    embedding_matrix_4), axis=1)
    # del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4
    embedding_matrix, _ = load_ggle(ggle_word_index, max_features, unk_uni)
    gc.collect()
    # with open('unknowns.pkl', 'wb') as f:
    #     pickle.dump({'glove': u1, 'wiki': u2, 'parag': u3, 'ggle': u4}, f)
    # print('Embedding:', embedding_matrix.shape)
    return embedding_matrix


def setup_emb(tr_X, max_features=50000, clean_num=2, unk_uni=True):
    tokenizer = Tokenizer(num_words=max_features, lower=False, filters='')
    tokenizer.fit_on_texts(tr_X)
    print('len(vocab)', len(tokenizer.word_index))
    embedding_matrix = load_all_embeddings(tokenizer, max_features=max_features,
                                           clean_num=clean_num, unk_uni=unk_uni)
    # np.save(embed_path, embedding_matrix)
    return tokenizer, embedding_matrix


def setup_emb_2(tr_X, max_features=50000, clean_num=2, unk_uni=True):
    full_tokenizer = Tokenizer(lower=False, filters='')
    full_tokenizer.fit_on_texts(tr_X)
    word2vec_path = data_path + 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    word2vec_dict = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    word2vec_tokenizer = get_vocab_by_embed(full_tokenizer, word2vec_dict)
    word2vec_matrix = get_embedding_matrix(word2vec_tokenizer, word2vec_dict)
    return word2vec_tokenizer, word2vec_matrix



puncts = ',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'


def clean_text(x, puncts=puncts):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def prepare_data(train_df, test_df, maxlen, max_features, trunc='pre',
                 lower=False, clean_num=2, unk_uni=True):
    train_df = train_df.copy()

    # lower
    if lower:
        train_df['question_text'] = train_df['question_text'].apply(lambda x: x.lower())
        test_df['question_text'] = test_df['question_text'].apply(lambda x: x.lower())

    # Clean the text
    train_df['question_text'] = train_df['question_text'].apply(
        lambda x: clean_text(x))
    test_df['question_text'] = test_df['question_text'].apply(
        lambda x: clean_text(x))

    # Clean numbers
    if clean_num == 1:
        train_df['question_text'] = train_df['question_text'].apply(
            lambda x: clean_numbers(x))
        test_df['question_text'] = test_df['question_text'].apply(
            lambda x: clean_numbers(x))

    # fill up the missing values
    train_df['question_text'] = train_df['question_text'].fillna('_##_')
    test_df['question_text'] = test_df['question_text'].fillna('_##_')

    train_X = train_df['question_text'].values
    test_X = test_df['question_text'].values

    tokenizer, embedding_matrix = setup_emb_2(train_X,
                                            max_features=max_features,
                                            clean_num=clean_num, unk_uni=unk_uni)

    tr_X_ids = tokenizer.texts_to_sequences(train_X)
    tr_X_padded = pad_sequences(tr_X_ids, maxlen=maxlen, truncating=trunc)
    test_X_ids = tokenizer.texts_to_sequences(test_X)
    test_X_padded = pad_sequences(test_X_ids, maxlen=maxlen, truncating=trunc)
    print(embedding_matrix.shape)
    embedding_matrix = torch.Tensor(embedding_matrix)
    return tr_X_padded, test_X_padded, embedding_matrix, tokenizer


ids_s = [list(range(300)), list(range(300, 600)),
         list(range(600, 900)), list(range(900, 1200))]

cols_s = [ids_s[0] + ids_s[1],
          ids_s[0] + ids_s[2],
          ids_s[1] + ids_s[2],
          ids_s[0] + ids_s[3],
          ids_s[1] + ids_s[3],
          ids_s[2] + ids_s[3]]

cols_one = ids_s[0]

n_embed = 1
embed_dim = n_embed * 300
n_models = 1

train_df = pd.read_csv(data_path + 'train.csv')
test_df = train_df.iloc[1000000:]
test_target = test_df.target.values

test_df = test_df.drop('target', axis=1)
train_df = train_df.iloc[:1000000]
# test_df = pd.read_csv(data_path + 'test.csv')
print('Train : ', train_df.shape)
print('Test : ', test_df.shape, test_target[:10])

obj = prepare_data(train_df, test_df, maxlen, max_features,
                   trunc=trunc, lower=lower, clean_num=clean_num, unk_uni=unk_uni)

train_X_padded, test_X_padded, embedding_matrix, _ = obj
train_y = train_df['target'].values

train_loader = get_dataloader(train_X_padded, train_y, batch_size=batch_size,
                              dtype_x=torch.long, dtype_y=torch.float, training=True,
                              drop_last=drop_last)
test_loader = get_dataloader(test_X_padded, y=None, batch_size=test_batch_size,
                             dtype_x=torch.long, training=False)


ema_n = int(train_df.shape[0] / (updates_per_epoch * batch_size))
test_pr = np.zeros((len(test_df), 1))
for i in range(n_models):
    # cols_in_use = cols_s[i % len(cols_s)]
    cols_in_use = cols_one
    model = GRUModel(n_vocab, embed_dim, proj_dim, rnn_dim, n_layers, bidirectional, dense_dim,
                     pretrained_embedding=embedding_matrix[:, cols_in_use],
                     fix_embedding=fix_embedding, padding_idx=0)
    if i == 0:
        print(model)
        print('#Trainable params', get_param_size(model))
    model.cuda()
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    optimizer = Adam([p for n, p in model.named_parameters() if p.requires_grad is True])
    ema = EMA(model, mu, n=ema_n)
    model.train()
    t2 = time.time()
    for e in range(epochs):
        epoch = e + 1
        loss = run_epoch(model, train_loader, optimizer, callbacks=[ema])
    print(f'n_model:{i + 1} {(time.time() - t2) / epochs:.1f}s/epoch')
    model.eval()
    model.gru.flatten_parameters()
    test_pr += model.predict_proba(test_loader)
    # ema.set_weights(ema_model)
    # To avoid RuntimeWarning:
    #   RNN module weights are not part of
    #   single contiguous chunk of memory. This means they need to
    #   be compacted at every call, possibly greatly increasing memory usage.
    #   To compact weights again call flatten_parameters().
    # ema_model.gru.flatten_parameters()
    t3 = time.time()
    # test_pr += ema_model.predict_proba(test_loader)
    print(f'{time.time() - t3:.1f}s')


test_pr /= n_models
test_pr = (test_pr > threshold).astype(int)
out_df = pd.DataFrame({"qid": test_df["qid"].values})
out_df['prediction'] = test_pr
out_df['target'] = test_target
out_df.to_csv("submission.csv", index=False)
print(f'Done:{time.time() - t0:.1f}s')
print(f1_score(out_df['target'].values, out_df['prediction'].values))
