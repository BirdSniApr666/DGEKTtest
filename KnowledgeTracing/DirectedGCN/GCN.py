import torch.nn as nn
import torch.nn.functional as F
import torch
from KnowledgeTracing.DirectedGCN.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)


    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.relu(self.gc2(x1, adj))
        return x2

    def forward_personal(self, x, adj, nodes, rows, neighbors, counts, alpha):
        """Use posterior transition probabilities in the second message-passing layer.

        For the immutable train graph P and one prefix row of personal counts c,
        the last propagation uses (alpha * P + c) / (alpha + sum(c)).
        The first propagation is shared so only the queried prefix rows need expanding.
        """
        hidden = F.relu(self.gc1(x, adj))
        messages = hidden @ self.gc2.weight
        population = torch.sparse.mm(adj, messages)
        prior = population[nodes.clamp_min(0)].flatten(0, 1)
        counts = counts.to(messages.dtype)
        totals = counts.new_zeros(prior.size(0)).index_add_(0, rows, counts)
        local = torch.zeros_like(prior).index_add_(0, rows, messages[neighbors] * counts[:, None])
        # In the absence of personal evidence this expression is exactly the prior.
        denominator = (alpha + totals[:, None]).masked_fill(totals[:, None] == 0, 1.0)
        mixed = prior + (local - totals[:, None] * prior) / denominator
        if self.gc2.bias is not None:
            mixed = mixed + self.gc2.bias
        mixed = F.relu(mixed).reshape(*nodes.shape, messages.size(-1))
        return mixed.masked_fill(nodes.unsqueeze(-1) < 0, 0)
