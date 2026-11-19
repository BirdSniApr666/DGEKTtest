"""Indexed DGEKT inputs and causal, student-specific transition counts."""
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from KnowledgeTracing.data.split import read_blocks


def _student_chunks(nodes, max_step, personal):
    # These counts live for exactly one student, across all of their chunks.
    incoming, outgoing = defaultdict(Counter), defaultdict(Counter)
    previous = None
    for start in range(0, len(nodes), max_step):
        chunk = np.full(max_step, -1, dtype=np.int64)
        observed = nodes[start:start + max_step]
        chunk[:len(observed)] = observed
        sample = {'nodes': chunk}
        if personal:
            in_edges, out_edges = [], []
            for step, current in enumerate(observed):
                # The prediction at this position has already seen current.
                # Its outcome is used only to predict the following question.
                if previous is not None:
                    outgoing[previous][current] += 1
                    incoming[current][previous] += 1
                in_edges.extend((step, other, count)
                                for other, count in incoming.get(current, {}).items())
                out_edges.extend((step, other, count)
                                 for other, count in outgoing.get(current, {}).items())
                previous = current
            # Snapshot only nonzero neighbors of the current node at each step.
            # Storing numeric arrays avoids retaining counters or a dense graph.
            sample['in'] = np.asarray(in_edges, dtype=np.int32).reshape(-1, 3)
            sample['out'] = np.asarray(out_edges, dtype=np.int32).reshape(-1, 3)
        yield sample


class PersonalDataset(Dataset):
    """Build each student's causal prefixes before the DataLoader shuffles chunks.

    Node IDs are zero-based, with incorrect responses in the second half of
    the node table. A padding position has ID -1 and no incident edges.
    With personal=False this supplies the same indexed chunks without edges.
    """

    def __init__(self, path, max_step, num_questions, personal=True):
        if max_step < 1 or num_questions < 1:
            raise ValueError('max_step and num_questions must be positive')
        self.samples = []
        for block in read_blocks(path):
            questions, answers = block['questions'], block['answers']
            if len(questions) != block['seq_len'] or len(answers) != len(questions):
                raise ValueError(f"Incorrect sequence length in {path}, block {block['block_id']}")
            if any(q < 1 or q > num_questions for q in questions):
                raise ValueError(f"Question ID outside 1..{num_questions} in {path}, block {block['block_id']}")
            if any(a not in (0, 1) for a in answers):
                raise ValueError(f"Expected binary answers in {path}, block {block['block_id']}")
            nodes = [q - 1 + (1 - a) * num_questions for q, a in zip(questions, answers)]
            self.samples.extend(_student_chunks(nodes, max_step, personal))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_personal(samples):
    """Return CPU tensors; row b * max_step + t addresses a single prefix."""
    nodes = torch.from_numpy(np.stack([s['nodes'] for s in samples]))
    batch = {'nodes': nodes}
    if 'in' not in samples[0]:
        return batch

    for direction in ('in', 'out'):
        edges = [s[direction] for s in samples]
        combined = np.concatenate(edges)
        row_offsets = np.repeat(np.arange(len(samples)) * nodes.shape[1],
                                [len(edge_list) for edge_list in edges])
        batch[direction + '_rows'] = torch.from_numpy(combined[:, 0] + row_offsets).long()
        batch[direction + '_neighbors'] = torch.from_numpy(combined[:, 1].astype(np.int64))
        batch[direction + '_counts'] = torch.from_numpy(combined[:, 2].astype(np.float32))
    return batch
