# -*- coding: utf-8 -*-
"""基线复现：加载仓库自带的 save2017model.pkl，在原始 test 集上评估。

不改动仓库任何既有文件，仅用于确认环境可用性与论文数字对齐基准。

用法：
    cd /workspace && python scripts/check_baseline.py
"""
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

import torch  # noqa: E402

from KnowledgeTracing.Constant import Constants as C  # noqa: E402
from KnowledgeTracing.data.dataloader import getLoader  # noqa: E402
from KnowledgeTracing.evaluation import eval  # noqa: E402


def main():
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0))
    print('Dataset:', C.DATASET, '| Questions:', C.NUM_OF_QUESTIONS)

    _, _, test_loaders = getLoader(C.DATASET)
    loss_func = eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)

    ckpt_path = C.REF_CKPT_PATH
    print('Loading checkpoint:', ckpt_path)
    # torch >= 2.6 默认 weights_only=True，整模型 pickle 需显式关闭
    model = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.eval()

    with torch.no_grad():
        auc, acc = eval.test_epoch(model, test_loaders, loss_func, device)

    print('=' * 60)
    print('BASELINE  AUC = %.6f   ACC = %.6f' % (auc, acc))
    print('PAPER     AUC = 0.775400 (ASSIST17, Table 3 of DGEKT)')
    print('=' * 60)


if __name__ == '__main__':
    main()
