# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : load_data.py
# @Project: GOODKT
# @Comment :
import numpy as np
import scipy.sparse as sp
from KnowledgeTracing.Constant import Constants as C
import tqdm
import itertools
import torch


def get_adj(train_path=None):
    """构建有向转移图（DTG）的出入邻接矩阵。

    Args:
        train_path: 用于统计交互转移的训练集 CSV。默认取 ``C.TRAIN_PATH``，
            即 block 级切分后的训练集；**绝不能传入完整训练集**，否则验证集
            的交互会泄漏进图结构中。
    """
    q = C.NUM_OF_QUESTIONS
    # Identity entries provide the original one self-loop per node and also
    # make rows for unobserved interactions well defined.
    sources = [np.arange(2 * q)]
    targets = [sources[0]]
    path = train_path or C.TRAIN_PATH

    with open(path, 'r', encoding='UTF-8-sig') as train:
        for length, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='Generate adjacency matrix',
                                              mininterval=2):
            if ans is None:
                raise ValueError('Incomplete four-line student record in ' + str(path))
            length = int(length.strip().strip(','))
            questions = np.array(ques.strip().strip(',').split(','), dtype=np.int64)
            answers = np.array(ans.strip().strip(',').split(','), dtype=np.int64)
            if (len(questions) != length or len(answers) != length or
                    (questions < 1).any() or (questions > q).any() or
                    ((answers != 0) & (answers != 1)).any()):
                raise ValueError('Invalid question, answer, or sequence length in ' + str(path))
            nodes = questions - 1 + (answers == 0) * q
            sources.append(nodes[:-1])
            targets.append(nodes[1:])
    rows, cols = np.concatenate(sources), np.concatenate(targets)
    transitions = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * q, 2 * q)).tocsr()
    return (sparse_mx_to_torch_sparse_tensor(normalize(transitions)),
            sparse_mx_to_torch_sparse_tensor(normalize(transitions.T)))


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()
