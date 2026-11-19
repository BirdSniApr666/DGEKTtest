# -*- coding: utf-8 -*-
"""Export aligned next-question probabilities. Step 1 predicts the chunk's second answer."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.split import iter_chunks
from KnowledgeTracing.evaluation import eval as kt_eval
from KnowledgeTracing.evaluation import pipeline

COLUMNS = ['split', 'sample_idx', 'block_id', 'chunk_id', 'step',
           'question_id', 'concept_id', 'y_true',
           'prob_cahg', 'prob_dtg', 'prob_ens', 'prob_mean', 'y_pred']

logger = pipeline.logger


def _load_model(device: torch.device, ckpt: str):
    model = pipeline.build_model(device)
    model, _, _ = pipeline.load_checkpoint(ckpt, model)
    model.eval()
    return model


def export_split(model, loader, loss_func, device, data_path: str,
                 out_path: str, split_name: str) -> Dict:
    """在单个 split 上导出逐时间步预测。"""
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError('Prediction export requires a sequential loader to align CSV rows')
    auc, acc, details = kt_eval.test_epoch(model, loader, loss_func, device, collect=True)

    chunks = list(iter_chunks(data_path, C.MAX_STEP))
    if len(chunks) != len(loader.dataset):
        raise AssertionError(
            f'chunk 数不一致：iter_chunks={len(chunks)} vs dataset={len(loader.dataset)} '
            f'(path={data_path})')

    seen = {item['sample_idx'] for item in details}
    if seen != set(range(len(chunks))):
        raise AssertionError('导出的样本索引未完整覆盖数据集，预测对齐已失效')

    blocks, chunk_ids, steps, sample_ids = [], [], [], []
    qids, cids, ys = [], [], []
    pcs, pts, pes, mean_probs = [], [], [], []
    padded_chunks = 0

    for item in details:
        idx = item['sample_idx']
        length = int(item['valid_len'])
        chunk = chunks[idx]

        step_idx = item['steps'].numpy()
        answers = np.asarray(chunk['answers'], dtype=np.int64)
        expected_steps = np.flatnonzero((answers[:-1] >= 0) & (answers[1:] >= 0)) + 1
        if not np.array_equal(step_idx, expected_steps):
            raise AssertionError(f'Sample {idx}: predicted steps disagree with source CSV')
        if not length:
            padded_chunks += 1
            continue
        q = np.asarray(chunk['questions'], dtype=np.int64)[step_idx]
        c = np.asarray(chunk['concepts'], dtype=np.int64)[step_idx]
        y = answers[step_idx]
        if not np.array_equal(item['a'].numpy(), y) or (q <= 0).any():
            raise AssertionError(f'Sample {idx}: targets disagree with source CSV')
        n = length
        blocks.extend([chunk['block_id']] * n)
        chunk_ids.extend([chunk['chunk_id']] * n)
        steps.extend(step_idx.tolist())
        sample_ids.extend([idx] * n)
        qids.append(q)
        cids.append(c)
        ys.append(y)
        pcs.append(item['pc'].numpy())
        pts.append(item['pt'].numpy())
        pes.append(item['penm'].numpy())
        mean_probs.append(item['p_mean'].numpy())

    if not qids:
        raise AssertionError(f'{split_name} 没有可导出的有效时间步')

    prob_c = np.concatenate(pcs)
    prob_t = np.concatenate(pts)
    prob_e = np.concatenate(pes)
    y_true = np.concatenate(ys)

    prob_mean = np.concatenate(mean_probs)
    frame = pd.DataFrame({
        'split': split_name,
        'sample_idx': sample_ids,
        'block_id': blocks,
        'chunk_id': chunk_ids,
        'step': steps,
        'question_id': np.concatenate(qids),
        'concept_id': np.concatenate(cids),
        'y_true': y_true,
        'prob_cahg': prob_c,
        'prob_dtg': prob_t,
        'prob_ens': prob_e,
        'prob_mean': prob_mean,
        'y_pred': np.rint(prob_mean).astype(np.int64),
    }, columns=COLUMNS)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frame.to_csv(out_path, index=False)

    summary = {'split': split_name, 'csv': os.path.abspath(out_path),
               'rows': int(len(frame)), 'auc': float(auc), 'acc': float(acc),
               'excluded_padding_chunks': padded_chunks}
    logger.info('导出 %s: %s', split_name, summary)
    print(f'[export] {split_name}: {len(frame)} rows -> {out_path} (auc={auc:.6f})')
    return summary


def export_all(train_loader, valid_loader, test_loader, device,
               ckpt: Optional[str] = None, out_dir: Optional[str] = None) -> List[Dict]:
    """导出 valid 与 test 两个 split 的逐时间步预测。"""
    out_dir = out_dir or C.OUT_DIR
    ckpt = ckpt or os.path.join(C.CKPT_DIR, 'best.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'找不到模型权重: {ckpt}，请先执行 --stage train')

    model = _load_model(device, ckpt)
    loss_func = kt_eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)

    results = []
    if valid_loader is not None:
        results.append(export_split(model, valid_loader, loss_func, device,
                                    C.VALID_PATH,
                                    os.path.join(out_dir, 'predictions_valid.csv'),
                                    'valid'))
    results.append(export_split(model, test_loader, loss_func, device,
                                C.TEST_PATH,
                                os.path.join(out_dir, 'predictions_test.csv'),
                                'test'))
    return results
