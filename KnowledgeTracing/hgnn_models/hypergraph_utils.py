# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np
import torch
import scipy.sparse as sp


class FactorizedHypergraph(torch.nn.Module):
    """Multiply by H_norm @ H_norm.T without materializing a node-by-node matrix."""

    def __init__(self, incidence):
        super().__init__()
        self.register_buffer('incidence', incidence, persistent=False)
        self.register_buffer('incidence_t', incidence.transpose(0, 1).coalesce(), persistent=False)

    def matmul(self, features):
        return torch.sparse.mm(self.incidence, torch.sparse.mm(self.incidence_t, features))


def generate_factorized_G_from_H(H):
    """Unit-weight hyperedges: H_norm = D_vertex^(-1/2) H D_edge^(-1/2)."""
    incidence = sp.coo_matrix(H, dtype=np.float64, copy=True)
    if not np.isfinite(incidence.data).all() or (incidence.data < 0).any():
        raise ValueError('Hypergraph incidence weights must be finite and non-negative')
    vertex_degree = np.asarray(incidence.sum(axis=1)).ravel()
    edge_degree = np.asarray(incidence.sum(axis=0)).ravel()
    incidence.eliminate_zeros()
    # Every stored incidence has strictly positive vertex and edge degrees.
    incidence.data /= np.sqrt(vertex_degree[incidence.row] * edge_degree[incidence.col])
    return FactorizedHypergraph(sparse_mx_to_torch_sparse_tensor(incidence))


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1] # Number of columns of matrix = number of hyperedge
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H *W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    # np.mat 在 numpy>=2.0 已移除，np.asmatrix 是数值等价的替代（返回同一 np.matrix 视图）
    invDE = np.asmatrix(np.diag(np.power(DE, float(-1))))
    DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))
    W = np.asmatrix(np.diag(W))
    H = np.asmatrix(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(G))
        # G = torch.Tensor(G)
        return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.把一个sparse matrix转为torch中的稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()
