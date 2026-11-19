# -*- coding: utf-8 -*-
"""为已有的 checkpoint 补写题目初始特征 ques。

背景：改造前的 `Model.py` 把 ques 存成普通 tensor，不在 state_dict 内，
导致 20 轮训练得到的 best.pt / last.pt 缺少该字段。由于训练过程中 ques
从不更新（emb 不在 optimizer 里），用训练种子重放一次初始化即可精确还原，
效果等同于重新训练，但不必再跑 45 分钟。

用法：
    cd /workspace && python scripts/patch_checkpoints.py
"""
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

import torch

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.evaluation import pipeline


def main():
    torch.cuda.set_device(0)
    device = torch.device('cuda')

    pipeline.set_seed(C.SEED)
    model = pipeline.build_model(device)
    ques = model.ques.detach().cpu()
    print('rebuilt ques:', tuple(ques.shape), 'mean=%.6f' % float(ques.mean()))

    for name in ('best.pt', 'last.pt'):
        path = os.path.join(C.CKPT_DIR, name)
        if not os.path.exists(path):
            print('skip missing', path)
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['model'][pipeline.QUES_KEY] = ques
        state[pipeline.QUES_KEY] = ques
        torch.save(state, path)
        print('patched %s (epoch=%s, best_auc=%.6f)'
              % (path, state.get('epoch'), state.get('best_auc', 0.0)))


if __name__ == '__main__':
    main()
