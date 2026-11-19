# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : Constant.py
# @Project: GOODKT
# @Comment :
"""全局配置。

与原版相比的改动：
1. 所有路径改为基于 ``__file__`` 的绝对路径，不再依赖 ``cwd``；
2. 新增 train/valid/test 三份数据路径（valid 来自 block 级切分）；
3. 新增 ``apply_overrides()``，供命令行入口覆盖 epoch / lr / batch size 等；
4. 删除原 ``set_seed`` 中把 ``CUDA_VISIBLE_DEVICES`` 设成 ``'0,2'`` 的无效逻辑。
"""
import sys
import math
from pathlib import Path

# /workspace  —— KnowledgeTracing/Constant/Constants.py 向上两级
ROOT_DIR = Path(__file__).resolve().parents[2]

datasets = {
    'assist2009' : 'assist2009',
    'assist2012' : 'assist2012',
    'assist2017' : 'assist2017',
    'assistednet': 'assistednet',
}

# question number of each dataset
numbers = {
    'assist2009' : 16891,
    'assist2012' : 37125,
    'assist2017' : 3162,
    'assistednet': 10795,
}

skill = {
    'assist2009' : 101,
    'assist2012' : 188,
    'assist2017' : 102,
    'assistednet': 1676,   # 188
}


DATASET = datasets['assist2017']
NUM_OF_QUESTIONS = numbers['assist2017']
NUM_OF_CONCEPTS = skill['assist2017']
H = '2017'

MAX_STEP = 50
BATCH_SIZE = 128
LR = 0.001
EPOCH = 20
EMB = 256
HIDDEN = 128  # sequence model's
kd_loss = 5.00E-06
LAYERS = 1
TRANSITION = 'static'
PERSONAL_ALPHA = 5.0

# ---------------------------------------------------------------------------
# 路径（绝对路径，任意 cwd 均可运行）
# ---------------------------------------------------------------------------
Dpath = str(ROOT_DIR / 'Dataset')

# 训练随机种子（可被 CLI 覆盖），与切分种子分开以便单独控制
SEED = 216
SPLIT_SEED = 216
DATASET_DIR = ROOT_DIR / 'Dataset' / DATASET
SPLIT_DIR = DATASET_DIR / f'split_seed{SPLIT_SEED}'

# block 级 90/10 切分后的训练集与独立验证集
TRAIN_PATH = str(SPLIT_DIR / 'train.csv')
VALID_PATH = str(SPLIT_DIR / 'valid.csv')
# 原始训练集（不切分时的退化选项，仅用于对齐论文/复现历史结果）
FULL_TRAIN_PATH = str(DATASET_DIR / f'{DATASET}_pid_train.csv')
# 原始测试集，全程留白，只在最后评估一次
TEST_PATH = str(DATASET_DIR / f'{DATASET}_pid_test.csv')
# 题目-概念关联矩阵
H_PATH = str(ROOT_DIR / 'Dataset' / 'H' / f'{H}.csv')

# 产物目录
CKPT_DIR = str(ROOT_DIR / 'KnowledgeTracing' / 'model' / 'checkpoints')
OUT_DIR = str(ROOT_DIR / 'outputs')
LOG_DIR = str(ROOT_DIR / 'KnowledgeTracing' / 'evaluation' / 'log')

# 仓库自带的参照权重（论文代码 epoch 10，AUC≈0.7745），重训时不得覆盖
REF_CKPT_PATH = str(ROOT_DIR / 'KnowledgeTracing' / 'model' / 'save2017model.pkl')


def apply_overrides(**kwargs):
    """覆盖模块级超参，供 CLI 调用。

    仅接受已存在的配置名，避免拼写错误被静默忽略。
    """
    module = sys.modules[__name__]
    applied = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in module.__dict__:
            raise AttributeError(f'未知配置项: {key}')
        setattr(module, key, value)
        applied[key] = value
    return applied


def configure_experiment(dataset, transition, alpha, seed):
    """Set all dataset paths together and isolate named experiment artifacts."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and strictly positive')
    if transition not in ('static', 'personal'):
        raise ValueError(f'Unknown transition mode: {transition}')
    year = {'assist2009': '2009', 'assist2017': '2017'}[dataset]
    data_dir = ROOT_DIR / 'Dataset' / dataset
    split_dir = data_dir / f'split_seed{SPLIT_SEED}'
    variant = 'static' if transition == 'static' else f'personal_alpha{alpha}'
    run_dir = Path(dataset) / variant / f'seed{seed}'
    apply_overrides(
        DATASET=dataset, DATASET_DIR=data_dir, SPLIT_DIR=split_dir,
        NUM_OF_QUESTIONS=numbers[dataset], NUM_OF_CONCEPTS=skill[dataset], H=year,
        H_PATH=str(ROOT_DIR / 'Dataset' / 'H' / f'{year}.csv'),
        FULL_TRAIN_PATH=str(data_dir / f'{dataset}_pid_train.csv'),
        TRAIN_PATH=str(split_dir / 'train.csv'), VALID_PATH=str(split_dir / 'valid.csv'),
        TEST_PATH=str(data_dir / f'{dataset}_pid_test.csv'),
        TRANSITION=transition, PERSONAL_ALPHA=alpha, SEED=seed,
        OUT_DIR=str(ROOT_DIR / 'outputs' / run_dir),
        CKPT_DIR=str(ROOT_DIR / 'KnowledgeTracing' / 'model' / 'checkpoints' / run_dir),
    )
