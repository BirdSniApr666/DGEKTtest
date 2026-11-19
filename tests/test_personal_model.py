import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.DirectedGCN.load_data import get_adj
from KnowledgeTracing.hgnn_models.hypergraph_utils import generate_factorized_G_from_H
from KnowledgeTracing.model.Model import DKT


class PersonalModelTests(unittest.TestCase):
    def test_population_graph_counts_directions_and_student_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(C, 'NUM_OF_QUESTIONS', 2):
            path = Path(tmp) / 'train.csv'
            path.write_text('4\n1,2,1,2\n1,1,1,1\n1,0,1,0\n2\n2,2\n1,1\n1,1\n')
            outgoing, incoming = get_adj(path)
        raw = torch.eye(4)
        raw[0, 3], raw[3, 0], raw[1, 1] = 2, 1, 2
        torch.testing.assert_close(outgoing.to_dense(), raw / raw.sum(1, keepdim=True))
        torch.testing.assert_close(incoming.to_dense(), raw.T / raw.T.sum(1, keepdim=True))

    def test_factored_hypergraph_with_empty_and_weighted_edges(self):
        incidence = np.array([[1., 2., 0.], [1., 0., 0.], [0., 0., 0.]])
        H = sp.coo_matrix(incidence)
        graph = generate_factorized_G_from_H(H)
        features = torch.eye(3)
        left = torch.diag(torch.tensor([3. ** -.5, 1., 0.]))
        right = torch.diag(torch.tensor([.5, .5, 0.]))
        matrix = torch.from_numpy(incidence).float()
        torch.testing.assert_close(graph.matmul(features), left @ matrix @ right @ matrix.T @ left)
        np.testing.assert_equal(H.toarray(), incidence)

    @patch.object(C, 'NUM_OF_QUESTIONS', 2)
    @patch.object(C, 'EMB', 4)
    def test_second_layer_matches_explicit_posterior_and_cold_start(self):
        # build_model historically passes (outgoing, incoming), in that order.
        raw = torch.tensor([[1., 2., 0., 0.], [0., 1., 1., 0.],
                            [0., 0., 1., 0.], [1., 0., 0., 1.]])
        outgoing, incoming = raw / raw.sum(1, keepdim=True), raw.T / raw.T.sum(1, keepdim=True)
        args = (3, 1, torch.eye(4).to_sparse(), outgoing.to_sparse(), incoming.to_sparse())
        baseline = DKT(*args)
        model = DKT(*args, personal_alpha=5.)
        model.load_state_dict(baseline.state_dict(), strict=True)
        nodes = torch.tensor([[0, 1, 0, -1]])
        batch = {'nodes': nodes}
        for name in ('in', 'out'):
            batch[name + '_rows'] = torch.empty(0, dtype=torch.long)
            batch[name + '_neighbors'] = torch.empty(0, dtype=torch.long)
            batch[name + '_counts'] = torch.empty(0)
        onehot = torch.nn.functional.one_hot(nodes.clamp_min(0), 4).float()
        onehot[nodes < 0] = 0.
        for original, indexed, cold in zip(baseline(onehot), baseline({'nodes': nodes}), model(batch)):
            torch.testing.assert_close(original, indexed)
            torch.testing.assert_close(original, cold)

        batch.update(in_rows=torch.tensor([1, 2]), in_neighbors=torch.tensor([0, 1]),
                     in_counts=torch.tensor([1., 1.]), out_rows=torch.tensor([2]),
                     out_neighbors=torch.tensor([1]), out_counts=torch.tensor([1.]))
        inputs = []
        hook = model.rnn2.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        try:
            logits = model(batch)
        finally:
            hook.remove()
        expected = []
        for name, prior, graph in [('out', outgoing, model.net2), ('in', incoming, model.net1)]:
            posterior = prior[nodes[0, :3]] * 5
            posterior.index_put_((batch[name + '_rows'], batch[name + '_neighbors']),
                                 batch[name + '_counts'], accumulate=True)
            posterior /= posterior.sum(1, keepdim=True)
            hidden = torch.relu(prior @ (model.ques @ graph.gc1.weight) + graph.gc1.bias)
            expected.append(torch.relu(posterior @ (hidden @ graph.gc2.weight) + graph.gc2.bias))
        torch.testing.assert_close(inputs[0][0, :3], torch.cat(expected, dim=-1))
        torch.testing.assert_close(inputs[0][0, 3], torch.zeros(4))
        sum(head.sum() for head in logits).backward()
        self.assertTrue(torch.isfinite(model.net2.gc2.weight.grad).all())
