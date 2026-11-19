# -*- coding: utf-8 -*-
# @Time : 2021/12/23 13:43
# @Author : Yumo
# @File : Model.py
# @Project: GOODKT
# @Comment :

import math

from KnowledgeTracing.hgnn_models import HGNN
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
import torch
from KnowledgeTracing.DirectedGCN.GCN import GCN


class DKT(nn.Module):

    def __init__(self, hidden_dim, layer_dim, G, adj_in, adj_out, personal_alpha=None):
        super(DKT, self).__init__()
        if personal_alpha is not None and (not math.isfinite(personal_alpha) or personal_alpha <= 0):
            raise ValueError('personal_alpha must be a finite positive number')
        self.personal_alpha = personal_alpha
        '''initial feature'''
        emb_dim = C.EMB
        emb = nn.Embedding(2 * C.NUM_OF_QUESTIONS, emb_dim)
        # 原实现用普通 tensor 持有题目初始特征，导致它既不在 parameters 里
        # （训练中从不更新），也不在 state_dict 里（从 checkpoint 恢复时会按当时的
        # RNG 重新随机初始化，与训练时的模型不一致）。注册为 buffer 后数值不变，
        # 但能随 state_dict 一起保存/恢复。
        self.register_buffer('ques', emb.weight.detach())
        '''generate two graphs'''
        self.G = G
        self.adj_out = adj_out
        self.adj_in = adj_in
        '''DGCN'''
        self.net1 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))
        self.net2 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))

        '''HGCN'''
        self.net = HGNN(in_ch=C.EMB,
                        n_hid=C.EMB,
                        n_class=C.EMB)
        '''GRU'''
        self.rnn1 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        self.rnn2 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        '''kd'''
        self.fc_c = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_t = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_ensemble = nn.Linear(2 * hidden_dim, C.NUM_OF_QUESTIONS)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''SkillGraph: HGCN'''
        ques_h = self.net(self.ques, self.G)
        # Historical checkpoints call DKT(G, outgoing, incoming), reversing the
        # argument names. net2 uses successor rows and net1 uses predecessor rows.
        alpha = getattr(self, 'personal_alpha', None)
        if isinstance(x, dict):
            nodes = x['nodes']
            pad = nodes.unsqueeze(-1) < 0
            indices = nodes.clamp_min(0)
            x_h = ques_h[indices].masked_fill(pad, 0)
            if alpha is None:
                x_d = self._population_embeddings()[indices].masked_fill(pad, 0)
            else:
                outgoing = self.net2.forward_personal(
                    self.ques, self.adj_in, nodes, x['out_rows'], x['out_neighbors'], x['out_counts'], alpha)
                incoming = self.net1.forward_personal(
                    self.ques, self.adj_out, nodes, x['in_rows'], x['in_neighbors'], x['in_counts'], alpha)
                x_d = torch.cat((outgoing, incoming), dim=-1)
        else:
            if alpha is not None:
                raise ValueError('personal transitions require prefix counts and indexed interaction nodes')
            x_h = x.matmul(ques_h)
            x_d = x.matmul(self._population_embeddings())

        '''gru'''
        out_h, _ = self.rnn1(x_h)
        out_d, _ = self.rnn2(x_d)

        '''logits'''
        logit_c = self.fc_c(out_h)
        logit_t = self.fc_t(out_d)

        '''kd'''
        theta = self.sigmoid(self.w1(out_h) + self.w2(out_d))
        out_d = theta * out_d
        out_h = (1 - theta) * out_h
        emseble_logit = self.fc_ensemble(torch.cat([out_d, out_h], -1))

        return logit_c, logit_t, emseble_logit

    def _population_embeddings(self):
        outgoing = self.net2(self.ques, self.adj_in)
        incoming = self.net1(self.ques, self.adj_out)
        return torch.cat((outgoing, incoming), dim=-1)
