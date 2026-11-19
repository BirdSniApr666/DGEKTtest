# -*- coding: utf-8 -*-
"""DGEKT 训练 / 验证 / 测试管线。

设计要点：
- **best epoch 只由验证集决定**，测试集在训练全部结束后只评估一次；
- 每轮把 train_loss / valid_auc / valid_acc / lr / 耗时 / 是否 best 追加到
  ``outputs/history.csv``，可直接用于画收敛曲线；
- 支持断点续训：checkpoint 里同时保存 model、optimizer、epoch、best_auc 与 RNG 状态；
- 绝不覆盖仓库自带的 ``save2017model.pkl``，所有产物写入 ``model/checkpoints/`` 与 ``outputs/``。
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import optim as optima

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.DirectedGCN.load_data import get_adj
from KnowledgeTracing.hgnn_models import hypergraph_utils as hgut
from KnowledgeTracing.model.Model import DKT
from KnowledgeTracing.evaluation import eval as kt_eval

logger = logging.getLogger('main.pipe')

HISTORY_FIELDS = ['epoch', 'lr', 'train_loss', 'valid_auc', 'valid_acc',
                  'is_best', 'best_auc', 'elapsed_sec']


def set_seed(seed: int) -> None:
    """固定随机种子。

    注意：原实现会在此处把 ``CUDA_VISIBLE_DEVICES`` 改成 ``'0,2'``，在单卡机器上
    既无效又容易误解，已移除。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Optional[str] = None, name: str = 'main') -> logging.Logger:
    """初始化文件 + 控制台日志，避免重复添加 handler。"""
    log_dir = log_dir or C.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger(name)
    root.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        stamp = time.strftime('%Y_%m_%d_%H_%M_%S')
        handler = logging.FileHandler(os.path.join(log_dir, f'{stamp}_result.log'), encoding='utf-8')
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root.addHandler(handler)
    return root


def build_model(device: torch.device):
    """构建双图结构并实例化 DGEKT。

    注意：``DKT`` 的形参顺序是 ``(hidden_dim, layer_dim, G, adj_in, adj_out)``，
    而原论文代码的调用是 ``DKT(HIDDEN, LAYERS, G, adj_out, adj_in)``，二者相反。
    这是既有真实行为，**必须原样保留**，否则模型数值会改变。
    """
    incidence = pd.read_csv(C.H_PATH, header=None, dtype=np.float32).to_numpy()
    q, k = C.NUM_OF_QUESTIONS, C.NUM_OF_CONCEPTS
    if incidence.shape == (q, k):
        incidence = sparse.csr_matrix(incidence)
        incidence = sparse.block_diag([incidence, incidence], format='csr')
    elif incidence.shape != (2*q, 2*k):
        raise ValueError(f'{C.H_PATH}: expected {(q, k)} or {(2*q, 2*k)}, got {incidence.shape}')
    G = hgut.generate_factorized_G_from_H(incidence).to(device)
    adj_out, adj_in = get_adj(C.TRAIN_PATH)
    alpha = C.PERSONAL_ALPHA if C.TRANSITION == 'personal' else None
    model = DKT(C.HIDDEN, C.LAYERS, G, adj_out.to(device), adj_in.to(device),
                personal_alpha=alpha).to(device)
    # The population prior and CAHG are rebuilt from these files on each run.
    model.graph_config = {'dataset': C.DATASET, 'transition': C.TRANSITION, 'alpha': alpha,
                          'train_sha256': file_sha256(C.TRAIN_PATH), 'h_sha256': file_sha256(C.H_PATH)}
    return model


def file_sha256(path):
    digest = hashlib.sha256()
    # Text mode makes a run portable between CRLF and LF checkouts.
    with open(path, encoding='utf-8-sig') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), ''):
            digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


def build_optimizer(model):
    return optima.Adam(model.parameters(), lr=C.LR)


QUES_KEY = 'ques'


def save_checkpoint(path: str, model, optimizer, epoch: int, best_auc: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'graph_config': model.graph_config,
        # 兼容早期版本的 state_dict（当时 ques 尚未注册为 buffer），
        # 显式冗余保存一份，确保任何情况下都能恢复题目初始特征
        QUES_KEY: model.ques.detach().cpu(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,          # 已完成的最大 epoch（0 表示还没跑过）
        'best_auc': best_auc,
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        },
    }, path)


def load_checkpoint(path: str, model, optimizer=None):
    """加载 checkpoint；兼容仓库自带的整模型 pickle（含 ``model`` 属性检查）。"""
    # 用 cpu 作为 map_location：避免把 RNG state 等非参数张量也搬到 GPU，
    # load_state_dict 会自动对齐到模型所在设备。
    state = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = state.get('graph_config') if isinstance(state, dict) else None
    if saved_config is None and model.personal_alpha is not None:
        raise ValueError('Legacy checkpoints use the static graph; choose --transition static.')
    if saved_config is not None and saved_config != model.graph_config:
        raise ValueError(f'Checkpoint graph configuration mismatch: {saved_config} vs {model.graph_config}')
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        model_dict = state['model']
        missing, unexpected = model.load_state_dict(model_dict, strict=False)
        # 早期 checkpoint 的 state_dict 不含 ques（当时它是普通 tensor），
        # 从顶层冗余字段恢复
        if QUES_KEY not in model_dict:
            if QUES_KEY in state:
                model.ques = state[QUES_KEY].detach().to(model.ques.device)
            else:
                logger.warning('checkpoint 缺少题目初始特征 ques，将使用当前随机初始化；'
                               '请在 set_seed(C.SEED) 后重建模型以保证与训练时一致')
        if saved_config is not None and (missing or unexpected):
            raise ValueError(f'Checkpoint weights mismatch: missing={missing}, unexpected={unexpected}')
        if missing or unexpected:
            logger.warning('state_dict 非严格匹配: missing=%s unexpected=%s', missing, unexpected)
        if optimizer is not None and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        start_epoch = int(state.get('epoch', 0))
        best_auc = float(state.get('best_auc', 0.0))
        rng = state.get('rng')
        if rng is not None:
            try:
                torch.set_rng_state(rng['torch'].cpu() if torch.is_tensor(rng['torch']) else rng['torch'])
                if torch.cuda.is_available() and rng.get('cuda'):
                    torch.cuda.set_rng_state_all(rng['cuda'])
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
            except Exception as exc:  # pragma: no cover - RNG 恢复失败不应中断训练
                logger.warning('RNG 状态恢复失败，忽略: %s', exc)
        return model, start_epoch, best_auc
    # 整模型 pickle 回退（例如 save2017model.pkl）
    if hasattr(state, 'state_dict'):
        model.load_state_dict(state.state_dict(), strict=False)
        if hasattr(state, 'ques'):
            model.ques = state.ques.detach().to(model.ques.device)
        return model, 0, 0.0
    raise ValueError(f'无法识别的 checkpoint 格式: {path}')


def read_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def append_history(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in HISTORY_FIELDS})


def train(train_loader, valid_loader, device, epochs: int,
          resume: Optional[str] = None, ckpt_dir: Optional[str] = None,
          out_dir: Optional[str] = None,
          fallback_loader=None) -> Dict:
    """训练主循环，按验证集 AUC 保存最优模型。

    Args:
        fallback_loader: 未启用独立验证集时用于监控的 loader（通常为 test，
            仅用于对齐原论文代码行为，会造成信息泄漏，正式实验不应使用）。
    """
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    out_dir = out_dir or C.OUT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    history_path = os.path.join(out_dir, 'history.csv')
    best_path = os.path.join(ckpt_dir, 'best.pt')

    model = build_model(device)
    optimizer = build_optimizer(model)
    loss_func = kt_eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)

    start_epoch, best_auc = 0, 0.0
    if resume:
        model, start_epoch, best_auc = load_checkpoint(resume, model, optimizer)
        logger.info('从 %s 续训，已完成 %d 轮，best_auc=%.6f', resume, start_epoch, best_auc)
        print(f'[resume] from {resume}: epoch={start_epoch}, best_auc={best_auc:.6f}')

    if start_epoch == 0 and os.path.exists(history_path):
        os.remove(history_path)  # 全新训练时清空旧记录，避免曲线混淆

    monitor_loader = valid_loader if valid_loader is not None else fallback_loader
    monitor_name = 'valid' if valid_loader is not None else ('test(leak)' if fallback_loader else 'none')

    print(f'Train chunks: {len(train_loader.dataset)} / {len(train_loader)} batches')
    if monitor_loader is not None:
        print(f'Monitor[{monitor_name}] chunks: {len(monitor_loader.dataset)} / {len(monitor_loader)} batches')

    rows = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model, optimizer, train_loss = kt_eval.train_epoch(model, train_loader, optimizer, loss_func)

        if monitor_loader is not None:
            if monitor_name == 'test(leak)':
                logger.warning('未启用独立验证集，当前使用 test 选择 best epoch（存在信息泄漏）')
            val_auc, val_acc = kt_eval.test_epoch(model, monitor_loader, loss_func, device)
        else:
            val_auc, val_acc = float('nan'), float('nan')

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            save_checkpoint(best_path, model, optimizer, epoch + 1, best_auc)
        # 只保留 last.pt 与 best.pt：单个 state_dict 约 40MB，
        # 逐轮留存会让 60 轮训练产生数 GB 冗余产物
        save_checkpoint(os.path.join(ckpt_dir, 'last.pt'),
                        model, optimizer, epoch + 1, best_auc)

        row = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': round(train_loss, 6),
            'valid_auc': round(val_auc, 6) if val_auc == val_auc else '',
            'valid_acc': round(val_acc, 6) if val_acc == val_acc else '',
            'is_best': int(is_best),
            'best_auc': round(best_auc, 6),
            'elapsed_sec': round(time.time() - t0, 1),
        }
        rows.append(row)
        append_history(history_path, [row])

        logger.info('epoch %d | train_loss=%.6f | valid_auc=%.6f | valid_acc=%.6f | best=%d | %.1fs',
                    epoch + 1, train_loss, row['valid_auc'] or 0.0,
                    row['valid_acc'] or 0.0, is_best, row['elapsed_sec'])
        print('epoch %2d | loss %.6f | valid_auc %s | acc %s | best %s | %.1fs'
              % (epoch + 1, train_loss, row['valid_auc'], row['valid_acc'],
                 'YES' if is_best else 'no', row['elapsed_sec']))

    summary = {'epochs_run': len(rows), 'best_auc': best_auc,
               'best_ckpt': best_path, 'history': history_path}
    logger.info('训练结束: %s', summary)
    return summary


def test_once(test_loader, device, ckpt: Optional[str] = None,
              out_dir: Optional[str] = None, collect: bool = False) -> Dict:
    """加载 best 模型，在测试集上评估（训练结束后只调用一次）。"""
    out_dir = out_dir or C.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    ckpt = ckpt or os.path.join(C.CKPT_DIR, 'best.pt')

    model = build_model(device)
    model, _, best_auc = load_checkpoint(ckpt, model)
    loss_func = kt_eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)

    print(f'[{os.path.basename(ckpt)}] best_auc(from ckpt)={best_auc:.6f}')
    result = kt_eval.test_epoch(model, test_loader, loss_func, device, collect=collect)

    if collect:
        auc, acc, details = result
    else:
        auc, acc, details = result[0], result[1], None

    metrics = {
        'checkpoint': os.path.abspath(ckpt),
        'test_auc': auc,
        'test_acc': acc,
        'config': {
            'dataset': C.DATASET,
            'graph': model.graph_config,
            'num_questions': C.NUM_OF_QUESTIONS,
            'max_step': C.MAX_STEP,
            'batch_size': C.BATCH_SIZE,
            'lr': C.LR,
            'emb': C.EMB,
            'hidden': C.HIDDEN,
            'layers': C.LAYERS,
            'kd_loss': C.kd_loss,
            'epochs_cfg': C.EPOCH,
            'train_path': C.TRAIN_PATH,
            'valid_path': C.VALID_PATH,
            'test_path': C.TEST_PATH,
        },
        'test_chunks': len(test_loader.dataset),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    logger.info('测试结果: auc=%.6f acc=%.6f', auc, acc)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'config'},
                     indent=2, ensure_ascii=False))
    return {'metrics': metrics, 'details': details}
