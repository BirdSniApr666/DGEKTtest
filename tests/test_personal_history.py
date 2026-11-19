import tempfile
import unittest
from pathlib import Path

import torch

from KnowledgeTracing.data.personal import PersonalDataset, collate_personal


class PersonalHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'students.csv'

    def dataset(self, last_answer=1, personal=True):
        # Student 1 nodes: 0, 3, 0 | 3, 0, 0. Student 2 nodes: 3, 0.
        self.path.write_text(
            f'6\n1,2,1,2,1,1\n1,2,1,2,1,1\n1,0,1,0,1,{last_answer}\n'
            '2\n2,1\n2,1\n0,1\n', encoding='utf-8')
        return PersonalDataset(self.path, 3, 2, personal=personal)

    def test_direction_counts_chunk_boundary_and_student_reset(self):
        data = self.dataset()
        # Deliberately shuffle the samples, including a new, independent student.
        batch = collate_personal([data[1], data[2], data[0]])
        self.assertEqual(batch['nodes'].tolist(), [[3, 0, 0], [3, 0, -1], [0, 3, 0]])
        for direction, expected in (
            ('in', {(0, 0): 2, (1, 3): 2, (2, 3): 2, (2, 0): 1,
                    (4, 3): 1, (7, 0): 1, (8, 3): 1}),
            ('out', {(0, 0): 1, (1, 3): 2, (2, 3): 2, (2, 0): 1, (8, 3): 1}),
        ):
            rows = batch[direction + '_rows'].tolist()
            neighbors = batch[direction + '_neighbors'].tolist()
            counts = batch[direction + '_counts'].tolist()
            self.assertEqual(dict(zip(zip(rows, neighbors), counts)), expected)

    def test_future_response_cannot_change_earlier_prefixes(self):
        original = self.dataset(last_answer=1)
        altered = self.dataset(last_answer=0)
        a = collate_personal([original[0], original[1]])
        b = collate_personal([altered[0], altered[1]])
        self.assertTrue(torch.equal(a['nodes'].flatten()[:5], b['nodes'].flatten()[:5]))
        for direction in ('in', 'out'):
            for suffix in ('rows', 'neighbors', 'counts'):
                key = direction + '_' + suffix
                self.assertTrue(torch.equal(a[key][a[direction + '_rows'] < 5],
                                            b[key][b[direction + '_rows'] < 5]))

    def test_static_and_personal_alignment(self):
        personal = self.dataset()
        indexed = self.dataset(personal=False)
        a = collate_personal(list(personal))
        b = collate_personal(list(indexed))
        self.assertEqual(set(b), {'nodes'})
        self.assertTrue(torch.equal(a['nodes'], b['nodes']))
