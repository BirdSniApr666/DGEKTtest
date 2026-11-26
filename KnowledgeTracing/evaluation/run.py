# -*- coding: utf-8 -*-
"""DGEKT 统一命令行入口。   # Modified 11.26

用法示例::

    # 1) 按 block 级切分 train/valid（只需执行一次）
    python -m KnowledgeTracing.evaluation.run --stage split

    # 2) 20 轮初步训练（按 valid AUC 保存 best）
    python -m KnowledgeTracing.evaluation.run --stage train --epochs 20

    # 3) 延长训练：从 last.pt 续到 60 轮
    python -m KnowledgeTracing.evaluation.run --stage train --epochs 60 --resume last

    # 4) 一次性测试 + 逐时间步预测导出
    python -m KnowledgeTracing.evaluation.run --stage test
    python -m KnowledgeTracing.evaluation.run --stage export

    # 5) 一条命令串起 train -> test -> export
    python -m KnowledgeTracing.evaluation.run --stage all --epochs 20

相对路径已在 ``Constants`` 中改为绝对路径，因此可在任意 cwd 下执行。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import torch  # noqa: E402

from KnowledgeTracing.Constant import Constants as C  # noqa: E402
from KnowledgeTracing.data import split as split_mod  # noqa: E402
from KnowledgeTracing.data.dataloader import getTrainLoader, getValidLoader, getTestLoader  # noqa: E402
from KnowledgeTracing.evaluation import pipeline  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='DGEKT on ASSIST09 / ASSIST17: split / train / test / export')
    parser.add_argument('--dataset', choices=('assist2009', 'assist2017'), help='默认 assist2017')
    parser.add_argument('--transition', choices=('static', 'personal'),
                        help='static 为共享转移图，personal 为固定先验强度的个人前缀图')
    parser.add_argument('--cahg', choices=('original', 'difficulty'), default='original',
                        help='默认 original；difficulty 按题目难度将每条概念超边分为至多两组')
    parser.add_argument('--alpha', type=float, help='个人图群体先验的伪计数，默认 5')
    parser.add_argument('--h-file', help='题目—知识点关联矩阵 CSV 的路径')
    parser.add_argument('--device', choices=('cpu', 'cuda'), help='默认优先使用 CUDA')
    parser.add_argument('--stage', default='all',
                        choices=['split', 'train', 'test', 'export', 'all'],
                        help='执行阶段；all = train -> test -> export')
    parser.add_argument('--epochs', type=int, default=None, help='最大训练轮数（覆盖 Constants.EPOCH）')
    parser.add_argument('--lr', type=float, default=None, help='学习率（覆盖 Constants.LR）')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size（覆盖 Constants.BATCH_SIZE）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（覆盖 Constants.SPLIT_SEED）')
    parser.add_argument('--resume', default=None,
                        help='断点续训：checkpoint 路径，或填 "last" 使用 outputs 同级的 last.pt')
    parser.add_argument('--train-file', default=None, help='覆盖训练集 CSV 路径')
    parser.add_argument('--valid-file', default=None, help='覆盖验证集 CSV 路径')
    parser.add_argument('--test-file', default=None, help='覆盖测试集 CSV 路径')
    parser.add_argument('--ckpt', default=None, help='test/export 阶段使用的 checkpoint，默认 best.pt')
    parser.add_argument('--split-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--no-valid', action='store_true',
                        help='不使用验证集（退化为原论文代码行为：用 test 监控，仅用于快速对齐）')
    args = parser.parse_args(argv)
    if args.alpha is not None and (args.transition != 'personal' or not math.isfinite(args.alpha) or args.alpha <= 0):
        parser.error('--alpha requires --transition personal and a finite positive value')
    if args.transition == 'personal' and args.no_valid:
        parser.error('--transition personal requires an independent validation set')
    return args


def resolve_resume(value):
    if not value:
        return None
    if value in ('last', 'auto'):
        path = os.path.join(C.CKPT_DIR, 'last.pt')
    else:
        path = value
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到续训 checkpoint: {path}')
    return path


def main(argv=None):
    args = parse_args(argv)

    if args.dataset is not None or args.transition is not None or args.cahg == 'difficulty':
        C.configure_experiment(args.dataset or 'assist2017', args.transition or 'static',
                               5.0 if args.alpha is None else args.alpha,
                               216 if args.seed is None else args.seed, cahg=args.cahg)
    C.apply_overrides(
        EPOCH=args.epochs,
        LR=args.lr,
        BATCH_SIZE=args.batch_size,
        SEED=args.seed,
        TRAIN_PATH=args.train_file,
        VALID_PATH=args.valid_file,
        TEST_PATH=args.test_file,
        H_PATH=args.h_file,
    )

    if args.stage == 'split':
        meta = split_mod.split_blocks(C.FULL_TRAIN_PATH, str(C.SPLIT_DIR),
                                      ratio=args.split_ratio, seed=C.SPLIT_SEED)
        print(f'[split] train={meta["train"]["blocks"]} valid={meta["valid"]["blocks"]} '
              f'-> {C.SPLIT_DIR}')
        return

    needs_valid = not args.no_valid and args.stage in ('train', 'export', 'all')
    if needs_valid and not os.path.isfile(C.VALID_PATH):
        raise FileNotFoundError(f'Missing validation split {C.VALID_PATH}. Run --stage split for {C.DATASET} first.')
    pipeline.setup_logger()
    logger = pipeline.logger
    logger.info('argv=%s', sys.argv if argv is None else argv)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(0)
    pipeline.set_seed(C.SEED)

    logger.info('Dataset=%s | Transition=%s | CAHG=%s | Q=%d | MAX_STEP=%d | BATCH=%d | LR=%g | EPOCH=%d',
                C.DATASET, C.TRANSITION, C.CAHG, C.NUM_OF_QUESTIONS, C.MAX_STEP, C.BATCH_SIZE, C.LR, C.EPOCH)
    print(f'Device: {torch.cuda.get_device_name(0) if device.type == "cuda" else device}')
    print(f'Dataset: {C.DATASET} | Transition: {C.TRANSITION} | CAHG: {C.CAHG} | '
          f'Questions: {C.NUM_OF_QUESTIONS} | EPOCH: {C.EPOCH}')

    loader_options = {'indexed': True, 'personal': C.TRANSITION == 'personal'}
    train_loader = getTrainLoader(C.TRAIN_PATH, **loader_options) if args.stage in ('train', 'all') else None
    valid_loader = getValidLoader(C.VALID_PATH, **loader_options) if needs_valid else None
    needs_test = args.stage in ('test', 'export', 'all') or (args.stage == 'train' and args.no_valid)
    test_loader = getTestLoader(C.TEST_PATH, **loader_options) if needs_test else None

    if args.stage in ('train', 'all'):
        if valid_loader is None:
            logger.warning('未启用独立验证集，将退化为在 test 上监控 best epoch（信息泄漏，仅用于对齐论文）')
        summary = pipeline.train(train_loader, valid_loader, device, C.EPOCH,
                                 resume=resolve_resume(args.resume),
                                 fallback_loader=None if valid_loader is not None else test_loader)
        print(f'[train] done: {summary}')

    if args.stage in ('test', 'all'):
        ckpt = args.ckpt or os.path.join(C.CKPT_DIR, 'best.pt')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'找不到模型权重: {ckpt}，请先执行 --stage train')
        result = pipeline.test_once(test_loader, device, ckpt=ckpt)
        print(f'[test] AUC={result["metrics"]["test_auc"]:.6f} '
              f'ACC={result["metrics"]["test_acc"]:.6f}')

    if args.stage in ('export', 'all'):
        from KnowledgeTracing.evaluation import export as export_mod
        ckpt = args.ckpt or os.path.join(C.CKPT_DIR, 'best.pt')
        export_mod.export_all(train_loader, valid_loader, test_loader, device, ckpt=ckpt)


if __name__ == '__main__':
    main()
