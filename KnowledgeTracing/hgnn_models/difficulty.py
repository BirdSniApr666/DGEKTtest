"""Build concept-association hyperedges from training-only question difficulty."""
from itertools import zip_longest

import numpy as np
from scipy import sparse

from KnowledgeTracing.Constant.Constants import MAX_CLUSTERS, SMOOTHING


def _question_difficulty(train_path, num_questions):
    """Shrink each question's error rate toward the error rate of all training attempts."""
    attempts = np.zeros(num_questions, dtype=np.int64)
    errors = np.zeros_like(attempts)
    with open(train_path, encoding='utf-8-sig') as train:
        for block_id, (length, questions, _, answers) in enumerate(zip_longest(*[train] * 4)):
            if answers is None:
                raise ValueError(f'Incomplete student block {block_id} in {train_path}')
            length = int(length.strip().strip(','))
            ids = np.asarray(questions.strip().strip(',').split(','), dtype=np.int64) - 1
            responses = np.asarray(answers.strip().strip(',').split(','), dtype=np.int64)
            if (ids.size != length or responses.size != length or
                    np.any((ids < 0) | (ids >= num_questions)) or
                    np.any((responses != 0) & (responses != 1))):
                raise ValueError(f'Invalid training question, answer or length in block {block_id} of {train_path}')
            # add.at counts every occurrence, including repeated attempts within one block.
            np.add.at(attempts, ids, 1)
            np.add.at(errors, ids, 1 - responses)
    total_attempts = attempts.sum()
    if total_attempts == 0:
        raise ValueError(f'No training attempts in {train_path}')
    global_error = errors.sum() / total_attempts
    difficulty = (errors + SMOOTHING * global_error) / (attempts + SMOOTHING)
    difficulty[attempts == 0] = global_error
    return difficulty


def _two_means_cutoff(scores):
    """Return the first score in the harder group of the optimal one-dimensional split."""
    values = np.sort(scores)
    cuts = np.flatnonzero(values[1:] > values[:-1]) + 1
    if not cuts.size:
        return None
    # Minimizing within-group squared error is equivalent to maximizing between-group
    # variance. Center and scale first so tiny but distinct differences remain visible.
    scaled = (values - values[0]) / (values[-1] - values[0])
    sums = np.cumsum(scaled, dtype=np.longdouble)
    left_mean = sums[cuts - 1] / cuts
    right_size = values.size - cuts
    right_mean = (sums[-1] - sums[cuts - 1]) / right_size
    gains = cuts.astype(np.longdouble) * right_size * (right_mean - left_mean) ** 2
    # Mathematically tied gains can differ by roundoff. Favor the first (lowest) cut.
    tied = np.isclose(gains, gains.max(), rtol=16 * np.finfo(np.float64).eps, atol=0)
    return values[cuts[np.flatnonzero(tied)[0]]]


def build_difficulty_incidence(incidence, num_questions, num_concepts, train_path, clusters=MAX_CLUSTERS):
    """Split each original CAHG column independently, retaining every nonzero membership.

    A Q by K matrix is duplicated into the usual correct/incorrect blocks. A full
    2Q by 2K matrix retains its original supports and weights, even for asymmetric
    inputs. Output columns follow the input column order, easier group first. Empty
    groups are omitted. Concept counts combine the two original polarity columns:
    a concept is split if at least one of its columns split, empty if both were empty.
    """
    if clusters not in (1, MAX_CLUSTERS):
        raise ValueError(f'clusters must be 1 or {MAX_CLUSTERS}')
    if num_questions < 1 or num_concepts < 1:
        raise ValueError('num_questions and num_concepts must be positive')
    matrix = sparse.coo_matrix(incidence)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('Hypergraph incidence weights must be finite and non-negative')
    if matrix.shape == (num_questions, num_concepts):
        matrix = sparse.block_diag((matrix, matrix), format='csc')
    elif matrix.shape == (2 * num_questions, 2 * num_concepts):
        matrix = matrix.tocsc(copy=True)
    else:
        raise ValueError(f'Expected question/concept or signed interaction/concept incidence, got {matrix.shape}')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    difficulty = _question_difficulty(train_path, num_questions) if clusters > 1 else None
    group_counts = np.zeros(matrix.shape[1], dtype=np.int8)
    new_columns = np.empty(matrix.nnz, dtype=np.int64)
    next_column = 0
    for edge in range(matrix.shape[1]):
        start, stop = matrix.indptr[edge:edge + 2]
        if start == stop:
            continue
        memberships = new_columns[start:stop]
        memberships.fill(next_column)
        group_counts[edge] = 1
        if difficulty is not None:
            questions = matrix.indices[start:stop] % num_questions
            # Both polarities of the same question can occur in a custom edge.
            # Each question gets one vote when finding that edge's cluster boundary.
            cutoff = _two_means_cutoff(difficulty[np.unique(questions)])
            if cutoff is not None:
                memberships[difficulty[questions] >= cutoff] += 1
                group_counts[edge] = 2
        next_column += int(group_counts[edge])

    result = sparse.csr_matrix((matrix.data, (matrix.indices, new_columns)),
                               shape=(matrix.shape[0], next_column))
    concept_groups = group_counts.reshape(2, num_concepts).max(axis=0)
    stats = {
        'split_concepts': int(np.count_nonzero(concept_groups == 2)),
        'single_concepts': int(np.count_nonzero(concept_groups == 1)),
        'empty_concepts': int(np.count_nonzero(concept_groups == 0)),
        'num_hyperedges': result.shape[1],
        'nnz': result.nnz,
        'num_nodes': result.shape[0],
    }
    return result, stats
