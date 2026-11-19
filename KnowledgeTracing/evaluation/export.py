# -*- coding: utf-8 -*-
"""逐时间步预测导出。

对齐原理
--------
`lossFunc` 预测的是"第 t 步作答前的状态去答第 t+1 题"，因此对每个 chunk：

    step        = 1 .. valid_len        （chunk 内 0-based 索引即 step）
    question_id = chunk.questions[step]
    concept_id  = chunk.concepts[step]
    y_true      = chunk.answers[step]
    prob_*      = aligned_predictions 返回的 pc / pt / penm[step-1]

`valid_len` 由 ``aligned_predictions`` 的截断规则给出，与训练损失完全一致。
导出前会做三项断言（chunk 数、样本覆盖、标签逐位相等），任何不一致立即报错，
避免产出错位的预测文件。
"""
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
           'prob_cahg', 'prob_dtg', 'prob_ens', 'y_pred']

logger = pipeline.logger


def _load_model(device: torch.device, ckpt: str):
    model = pipeline.build_model(device)
    model, _, _ = pipeline.load_checkpoint(ckpt, model)
    model.eval()
    return model


def export_split(model, loader, loss_func, device, data_path: str,
                 out_path: str, split_name: str) -> Dict:
    """在单个 split 上导出逐时间步预测。"""
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
    pcs, pts, pes = [], [], []
    padded_chunks = 0

    for item in details:
        idx = item['sample_idx']
        length = int(item['valid_len'])
        chunk = chunks[idx]

        q = np.asarray(chunk['questions'][1:1 + length], dtype=np.int64)
        c = np.asarray(chunk['concepts'][1:1 + length], dtype=np.int64)
        y = np.asarray(chunk['answers'][1:1 + length], dtype=np.int64)
        a = item['a'].numpy().astype(np.int64)

        # 既有实现缺陷：当某 chunk 只有 1 个有效交互（seq_len % MAX_STEP == 1）时，
        # 对齐后的 pc 全为 0，截断循环找不到 pc[i] > 0，valid_len 会停在 MAX_STEP-1，
        # 于是 49 个 padding 步被当作"答错"样本计入损失与指标（论文原始代码同样如此）。
        # 这里不修改评测数值（保持与论文可比），只在导出时剔除这些 padding 步。
        mask = y >= 0
        if length <= 0 or not np.array_equal(a[mask], y[mask]):
            raise AssertionError(f'样本 {idx} 的真实标签与 chunk 不一致，预测对齐已失效')
        if mask.any() and (q[mask] <= 0).any():
            raise AssertionError(f'样本 {idx} 出现非法题号（{q[mask].min()}），chunk 解析有误')
        if not mask.any():
            padded_chunks += 1
            continue

        q, c, y = q[mask], c[mask], y[mask]
        step_idx = np.arange(1, length + 1)[mask]
        pc_row = item['pc'].numpy()[mask]
        pt_row = item['pt'].numpy()[mask]
        pe_row = item['penm'].numpy()[mask]

        n = int(mask.sum())
        blocks.extend([chunk['block_id']] * n)
        chunk_ids.extend([chunk['chunk_id']] * n)
        steps.extend(step_idx.tolist())
        sample_ids.extend([idx] * n)
        qids.append(q)
        cids.append(c)
        ys.append(y)
        pcs.append(pc_row)
        pts.append(pt_row)
        pes.append(pe_row)

    if not qids:
        raise AssertionError(f'{split_name} 没有可导出的有效时间步')

    prob_c = np.concatenate(pcs)
    prob_t = np.concatenate(pts)
    prob_e = np.concatenate(pes)
    y_true = np.concatenate(ys)

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
        'y_pred': (prob_e >= 0.5).astype(np.int64),
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
