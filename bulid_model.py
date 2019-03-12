import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import f1_score
from preprocess import get_train_data, split_train_eval, get_dataloader
from attention_cnn_model import model_build as attn_model_build
from gru_model import model_build as gru_model_build

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ModelConfig():
    def __init__(self, model_type):
        self.batch_size = 500
        if model_type == 'attn_cnn':
            self.heads = 6
            self.max_seq_len = 50
            self.cnn_filter = 128
            self.dropout = 0.2
            self.embed_freeze = True
        elif model_type == 'gru':
            self.maxlen = 50
            self.lower = False
            self.trunc = 'pre'
            self.max_features = 130000

            hidden_dim = 128

            # Embedding
            self.fix_embedding = True
            self.n_embed = 1
            self.embed_dim = self.n_embed * 300
            self.proj_dim = hidden_dim

            # GRU
            self.bidirectional = True
            self.n_layers = 1
            self.rnn_dim = hidden_dim

            self.dense_dim = 2 * self.rnn_dim if self.bidirectional else self.rnn_dim


def get_train_eval_data(config):
    batch_size = config.batch_size
    train_tuple, embedding_list = get_train_data()
    train_tuple, eval_tuple = split_train_eval(*train_tuple)
    train_loader = get_dataloader(*train_tuple, training=True, batch_size=batch_size, drop_last=True)
    test_loader = get_dataloader(*eval_tuple[:-1], training=False, batch_size=batch_size, drop_last=False)
    return train_loader, test_loader, embedding_list, eval_tuple[-1].data.numpy()


def run_epoch(model, dataloader, optimizer, callbacks=None,
              criterion=nn.BCEWithLogitsLoss(), verbose_step=1000):
    t1 = time.time()
    tr_loss = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        x_batch, m_batch, y_batch = batch

        model.zero_grad()
        outputs = model(x_batch, m_batch)
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


def eval_data(model, dataloader, y_eval, threshold=[0.3]):
    y_prob = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.cuda() for t in batch)
            x_batch, m_batch = batch
            outputs = model(x_batch, m_batch)
            outputs = torch.sigmoid(outputs)
            y_prob.append(outputs.data.cpu())
    y_prob = torch.cat(y_prob).data.numpy()

    for t in threshold:
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_eval, y_pred)
        print('eval f1 socre:', f1)


if __name__ == '__main__':
    seed_torch(2008)
    config = ModelConfig('gru')
    # config = ModelConfig('attn_cnn')
    train_loader, eval_loader, embedding_list, y_eval = get_train_eval_data(config)
    word2vec_matrix, word2vec_tokenizer = embedding_list[0]
    model = gru_model_build(word2vec_matrix, config)
    print(model)
    # model = attn_model_build(word2vec_matrix, config)
    # print(list(model.parameters())
    optimizer = Adam(model.parameters())
    model.cuda()
    for _ in range(12):
        model.train()
        loss = run_epoch(model, train_loader, optimizer)
        model.eval()
        eval_data(model, eval_loader, y_eval=y_eval, threshold=[0.25, 0.3,0.35,0.4,0.45,0.5])








