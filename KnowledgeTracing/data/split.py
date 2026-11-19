# -*- coding: utf-8 -*-
"""ASSIST17 数据块级切分与 chunk 枚举工具。

数据文件格式（纯文本，每 4 行一个学生序列，即一个 block）：
    第 1 行: seq_len              该学生真实交互次数
    第 2 行: questions            题号序列，逗号分隔
    第 3 行: concepts             概念序列，逗号分隔（原训练代码丢弃，此处仅用于导出）
    第 4 行: answers              答案序列，0/1

一个 block 在 `preprocess.DataReader` 中会被按 ``MAX_STEP`` 切分为多个 chunk，
因此切分 train/valid 必须在 **block 级** 进行，否则同一学生的前后 chunk 会被
拆到两侧造成泄漏。

本模块同时被数据切分（`split_blocks`）与预测导出（`iter_chunks`）复用，
保证两边对 chunk 的划分、pad 规则、顺序完全一致。
"""
from __future__ import annotations

import itertools
import json
import os
import random
from typing import Dict, Iterator, List

import numpy as np

BLOCK_LINES = 4


def _parse_int_line(line: str, name: str, block_id: int) -> List[int]:
    """把 ``1,2,3`` 形式的行解析为 int 列表。"""
    line = line.strip().strip(',')
    if not line:
        return []
    try:
        return [int(x) for x in line.split(',')]
    except ValueError as exc:  # pragma: no cover - 数据损坏时的兜底
        raise ValueError(f'block {block_id} 的 {name} 行无法解析为整数: {line[:80]!r}') from exc


def read_blocks(path: str) -> List[Dict]:
    """读取全部 block。

    Returns:
        每个元素形如 ``{block_id, seq_len, questions, concepts, answers}``。
    """
    blocks: List[Dict] = []
    with open(path, 'r', encoding='UTF-8-sig') as handle:
        # 与 preprocess.DataReader 完全一致的 4 行一组读取方式
        for block_id, group in enumerate(itertools.zip_longest(*[handle] * BLOCK_LINES)):
            len_line, ques_line, conc_line, ans_line = group
            # 文件行数不是 4 的倍数时，尾部不完整块直接丢弃
            if len_line is None or ans_line is None:
                break
            if not len_line.strip():
                continue

            questions = _parse_int_line(ques_line, 'questions', block_id)
            answers = _parse_int_line(ans_line, 'answers', block_id)
            concepts = _parse_int_line(conc_line, 'concepts', block_id)

            blocks.append(
                {
                    'block_id': block_id,
                    'seq_len': int(len_line.strip().strip(',')),
                    'questions': questions,
                    'concepts': concepts,
                    'answers': answers,
                }
            )
    return blocks


def write_blocks(path: str, blocks: List[Dict]) -> None:
    """按原始 4 行格式写出 block 列表。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        for blk in blocks:
            handle.write(f"{blk['seq_len']}\n")
            handle.write(','.join(str(x) for x in blk['questions']) + '\n')
            handle.write(','.join(str(x) for x in blk['concepts']) + '\n')
            handle.write(','.join(str(x) for x in blk['answers']) + '\n')


def iter_chunks(path: str, max_step: int) -> Iterator[Dict]:
    """按 ``preprocess.DataReader`` 的规则枚举 chunk。

    pad 与 reshape 规则与训练时逐位一致：先补 ``-1`` 到 ``max_step`` 的整数倍，
    再 reshape 成 ``[-1, max_step]``，块内顺序不变、块间按文件顺序排列。

    Yields:
        ``{block_id, chunk_id, questions, concepts, answers}``，其中
        questions/answers 已 pad 为长度 ``max_step``，concepts 同样 pad ``-1``。
    """
    for blk in read_blocks(path):
        questions = np.asarray(blk['questions'], dtype=int)
        answers = np.asarray(blk['answers'], dtype=int)
        concepts = np.asarray(blk['concepts'], dtype=int)

        length = blk['seq_len']
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        pad = np.zeros(mod, dtype=int) - 1

        questions = np.append(questions, pad).reshape([-1, max_step])
        answers = np.append(answers, pad).reshape([-1, max_step])
        # 概念序列长度与题号一致，按相同的 mod 补齐（源数据缺失时退化为 -1）
        if concepts.size == length:
            concepts = np.append(concepts, pad).reshape([-1, max_step])
        else:  # pragma: no cover - 正常数据不会走到这里
            concepts = np.full_like(questions, -1)

        for chunk_id in range(questions.shape[0]):
            yield {
                'block_id': blk['block_id'],
                'chunk_id': chunk_id,
                'questions': questions[chunk_id],
                'concepts': concepts[chunk_id],
                'answers': answers[chunk_id],
            }


def split_blocks(src: str, out_dir: str, ratio: float = 0.1, seed: int = 216) -> Dict:
    """按 block 级随机切分 ``src`` 为 train / valid 两份 CSV。

    Args:
        src: 原始训练集 CSV 路径。
        out_dir: 输出目录，会写入 ``train.csv``、``valid.csv``、``split_meta.json``。
        ratio: valid 占比。
        seed: 随机种子，与训练入口保持一致以便复现。
    """
    blocks = read_blocks(src)
    total = len(blocks)
    if total == 0:
        raise ValueError(f'未能从 {src} 解析出任何 block')

    n_valid = max(1, int(round(total * ratio)))
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    valid_idx = set(indices[:n_valid])

    train_blocks = [b for i, b in enumerate(blocks) if i not in valid_idx]
    valid_blocks = [b for i, b in enumerate(blocks) if i in valid_idx]

    train_path = os.path.join(out_dir, 'train.csv')
    valid_path = os.path.join(out_dir, 'valid.csv')
    write_blocks(train_path, train_blocks)
    write_blocks(valid_path, valid_blocks)

    def _stats(group: List[Dict]) -> Dict:
        records = sum(int(b['seq_len']) for b in group)
        return {
            'blocks': len(group),
            'records': records,
            'min_len': min((int(b['seq_len']) for b in group), default=0),
            'max_len': max((int(b['seq_len']) for b in group), default=0),
        }

    meta = {
        'source': os.path.abspath(src),
        'out_dir': os.path.abspath(out_dir),
        'split_unit': 'block (one student per 4 lines)',
        'seed': seed,
        'valid_ratio': ratio,
        'total_blocks': total,
        'train': _stats(train_blocks),
        'valid': _stats(valid_blocks),
        'total_records': sum(int(b['seq_len']) for b in blocks),
    }
    with open(os.path.join(out_dir, 'split_meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    return meta


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ASSIST17 block-level train/valid split')
    default_src = os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset',
                               'assist2017', 'assist2017_pid_train.csv')
    parser.add_argument('--src', default=os.path.normpath(default_src))
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=216)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.normpath(args.src)),
                                           f'split_seed{args.seed}')
    meta = split_blocks(args.src, out_dir, args.ratio, args.seed)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
