import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.evaluation import pipeline, run
from KnowledgeTracing.hgnn_models.difficulty import build_difficulty_incidence
from KnowledgeTracing.hgnn_models.hypergraph_utils import generate_factorized_G_from_H


def write_attempts(path, counts):
    """One synthetic student per question, with (incorrect, total) attempts."""
    with path.open('w', encoding='utf-8') as handle:
        for question, (incorrect, total) in enumerate(counts, start=1):
            if total:
                answers = ['0'] * incorrect + ['1'] * (total - incorrect)
                handle.write(f'{total}\n')
                handle.write(','.join([str(question)] * total) + '\n')
                handle.write(','.join(['1'] * total) + '\n')
                handle.write(','.join(answers) + '\n')


class DifficultyCAHGTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.train = Path(directory.name) / 'train.csv'

    def test_optimal_split_is_local_and_preserves_multi_concept_weights_and_signs(self):
        write_attempts(self.train, [(0, 20), (1, 20), (2, 20), (19, 20), (20, 20)])
        raw = np.array([[2, 1, 0, 0],
                        [1, 0, 3, 0],
                        [1, 4, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]], dtype=np.float32)
        # The first three questions form the easy group of concept 1.
        # Questions 1 and 3 also share concept 2 and split into separate groups there.
        expected = np.array([[2, 0, 1, 0, 0],
                             [1, 0, 0, 0, 3],
                             [1, 0, 0, 4, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0]], dtype=np.float32)
        signed = sp.block_diag([expected, expected], format='csr')
        for incidence in (raw, sp.block_diag([raw, raw], format='csr')):
            with self.subTest(shape=incidence.shape):
                result, stats = build_difficulty_incidence(incidence, 5, 4, self.train)
                self.assertTrue(sp.isspmatrix_csr(result))
                self.assertEqual(result.shape, signed.shape)
                self.assertEqual((result != signed).nnz, 0)
                self.assertEqual(result.nnz, 2 * np.count_nonzero(raw))
                self.assertEqual(stats['split_concepts'], 2)
                self.assertEqual(stats['single_concepts'], 1)
                self.assertEqual(stats['empty_concepts'], 1)

    def test_all_attempt_smoothing_and_unseen_question_prior(self):
        write_attempts(self.train, [(0, 1), (1, 1), (2, 2), (30, 50), (0, 0)])
        # The training error prior is 33/54. With 20 prior attempts, the
        # difficulties are .582, .630, .646, .603, .611, in question order.
        result, _ = build_difficulty_incidence(sp.csr_matrix(np.ones((5, 1))), 5, 1, self.train)
        expected = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])
        np.testing.assert_array_equal(result.toarray(), sp.block_diag([expected, expected]).toarray())

    def test_equal_scores_never_split_and_tied_optimum_is_deterministic(self):
        write_attempts(self.train, [(0, 20), (10, 20), (20, 20)])
        raw = np.ones((3, 1))
        result, _ = build_difficulty_incidence(raw, 3, 1, self.train)
        # At .25, .5 and .75, both boundaries have equal SSE.
        np.testing.assert_array_equal(result[:3, :2].toarray(), [[1, 0], [0, 1], [0, 1]])

        write_attempts(self.train, [(1, 2), (5, 10), (0, 0)])
        result, stats = build_difficulty_incidence(raw, 3, 1, self.train)
        self.assertEqual(stats['split_concepts'], 0)
        self.assertEqual(result.shape, (6, 2))
        np.testing.assert_array_equal(result.toarray(), sp.block_diag([raw, raw]).toarray())

    def test_single_cluster_matches_original_propagation_for_asymmetric_signed_input(self):
        write_attempts(self.train, [(0, 1), (1, 1), (0, 0)])
        correct = np.array([[1, 3, 0], [1, 0, 0], [0, 0, 0]])
        incorrect = np.array([[0, 0, 5], [2, 0, 0], [0, 0, 0]])
        signed = sp.block_diag([correct, incorrect], format='csr')
        result, _ = build_difficulty_incidence(signed, 3, 3, self.train, clusters=1)
        features = torch.eye(6)
        original = generate_factorized_G_from_H(signed).matmul(features)
        clustered = generate_factorized_G_from_H(result).matmul(features)
        torch.testing.assert_close(clustered, original)
        self.assertTrue(torch.isfinite(clustered).all())

    def test_cli_routes_difficulty_to_separate_outputs_and_checkpoints(self):
        root = self.train.parent
        split_meta = {'train': {'blocks': 1}, 'valid': {'blocks': 1}}
        with patch.dict(vars(C)), patch.object(C, 'ROOT_DIR', root), \
                patch.object(run.split_mod, 'split_blocks', return_value=split_meta):
            for flags, variant in [(['--cahg', 'difficulty'], 'static_cahg_difficulty_k2_m20'),
                                   (['--dataset', 'assist2017', '--cahg', 'original'], 'static')]:
                with self.subTest(flags=flags):
                    run.main(['--stage', 'split', *flags])
                    self.assertEqual(C.CAHG, flags[-1])
                    run_path = Path('assist2017') / variant / 'seed216'
                    self.assertEqual(Path(C.OUT_DIR), root / 'outputs' / run_path)
                    self.assertEqual(Path(C.CKPT_DIR),
                                     root / 'KnowledgeTracing' / 'model' / 'checkpoints' / run_path)

    def test_model_build_and_checkpoint_compatibility(self):
        write_attempts(self.train, [(0, 3), (3, 3)])
        incidence_file = self.train.with_name('incidence.csv')
        np.savetxt(incidence_file, [[1], [1]], fmt='%d', delimiter=',')
        with patch.multiple(C, TRAIN_PATH=str(self.train), H_PATH=str(incidence_file),
                            VALID_PATH=str(self.train.with_name('unavailable-valid.csv')),
                            TEST_PATH=str(self.train.with_name('unavailable-test.csv')),
                            FULL_TRAIN_PATH=str(self.train.with_name('unavailable-full.csv')),
                            NUM_OF_QUESTIONS=2, NUM_OF_CONCEPTS=1, EMB=4, HIDDEN=3,
                            TRANSITION='static', CAHG='original'):
            original = pipeline.build_model(torch.device('cpu'))
            with patch.object(C, 'CAHG', 'difficulty'):
                difficulty = pipeline.build_model(torch.device('cpu'))

        torch.testing.assert_close(original.adj_in.to_dense(), difficulty.adj_in.to_dense())
        torch.testing.assert_close(original.adj_out.to_dense(), difficulty.adj_out.to_dense())
        features = torch.eye(4)
        self.assertFalse(torch.allclose(original.G.matmul(features), difficulty.G.matmul(features)))
        self.assertEqual(set(original.graph_config), {'dataset', 'transition', 'alpha',
                                                       'train_sha256', 'h_sha256'})
        self.assertEqual(difficulty.graph_config, {
            **original.graph_config, 'cahg': 'difficulty',
            'cahg_estimator': 'all_attempts_smoothed_error_v1',
            'cahg_clustering': 'optimal_1d_sse', 'cahg_clusters': 2, 'cahg_smoothing': 20,
        })
        path = self.train.with_name('synthetic.pt')
        for source, wrong in ((original, difficulty), (difficulty, original)):
            torch.save({'model': source.state_dict(), 'graph_config': source.graph_config,
                        'epoch': 3}, path)
            self.assertEqual(pipeline.load_checkpoint(path, source)[1], 3)
            with self.assertRaises(ValueError):
                pipeline.load_checkpoint(path, wrong)
        torch.save({'model': original.state_dict(), 'epoch': 2}, path)
        self.assertEqual(pipeline.load_checkpoint(path, original)[1], 2)
        with self.assertRaises(ValueError):
            pipeline.load_checkpoint(path, difficulty)
