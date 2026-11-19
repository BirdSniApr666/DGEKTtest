import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.personal import PersonalDataset, collate_personal
from KnowledgeTracing.evaluation.eval import lossFunc
from KnowledgeTracing.evaluation.export import export_split


class FixedHeads(torch.nn.Module):
    def forward(self, batch):
        shape = (*batch['nodes'].shape, 2)
        graph_logits = torch.tensor([-2., 2.]).expand(shape)
        return graph_logits, graph_logits, -graph_logits


class PersonalEvaluationTests(unittest.TestCase):
    def test_legacy_and_indexed_alignment_and_gradients(self):
        nodes = torch.tensor([[0, 3, 1], [2, -1, -1]])
        dense = torch.nn.functional.one_hot(nodes.clamp_min(0), num_classes=4).float()
        dense[nodes < 0] = 0
        with patch.object(C, 'NUM_OF_QUESTIONS', 2):
            criterion = lossFunc(4, 3, 'cpu')
            logits = [torch.arange(12.).reshape(2, 3, 2).requires_grad_() for _ in range(3)]
            indexed = criterion(*logits, {'nodes': nodes})
            legacy = criterion(*logits, dense)
            for a, b in zip(indexed, legacy):
                torch.testing.assert_close(a, b)
            predictions, labels = indexed[2:]
            torch.testing.assert_close(predictions, torch.sigmoid(torch.tensor([1., 3.])))
            torch.testing.assert_close(labels, torch.tensor([0., 1.]))
            (indexed[0] + indexed[1]).backward()
            self.assertTrue(all(t.grad.isfinite().all() for t in logits))
            self.assertTrue(all(t.grad[1].count_nonzero() == 0 for t in logits))

    def test_export_matches_metrics_and_singleton_tail(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(C, 'MAX_STEP', 3), \
                patch.object(C, 'NUM_OF_QUESTIONS', 2):
            path = Path(directory) / 'students.csv'
            path.write_text('4\n1,2,1,2\n1,2,1,2\n1,1,0,1\n'
                            '2\n2,1\n2,1\n1,0\n', encoding='utf-8')
            dataset = PersonalDataset(path, 3, 2)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_personal)
            csv = Path(directory) / 'predictions.csv'
            results = export_split(FixedHeads(), loader, lossFunc(4, 3, 'cpu'),
                                   'cpu', str(path), str(csv), 'test')
            rows = pd.read_csv(csv)
            self.assertEqual(rows[['sample_idx', 'step', 'question_id', 'y_true']].values.tolist(),
                             [[0, 1, 2, 1], [0, 2, 1, 0], [2, 1, 1, 0]])
            self.assertEqual(results['excluded_padding_chunks'], 1)
            self.assertAlmostEqual(roc_auc_score(rows.y_true, rows.prob_mean), results['auc'])
            self.assertAlmostEqual(accuracy_score(rows.y_true, rows.y_pred), results['acc'])
            self.assertTrue((rows.y_pred != (rows.prob_ens >= .5)).all())
