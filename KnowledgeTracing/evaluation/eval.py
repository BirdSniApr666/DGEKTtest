# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : eval.py
# @Project: GOODKT
# @Comment :
"""损失与评估。

与原版差异：
1. 把「每步预测 ↔ 下一步作答」的对齐与 pad 截断逻辑抽成 ``aligned_predictions``，
   训练 / 验证 / 测试 / 预测导出共用同一份实现，避免 DRY 被破坏；
2. ``lossFunc.forward`` 的损失数值与聚合顺序保持不变；
3. ``test_epoch`` 新增 ``collect=True``：额外返回逐样本明细（三路概率 + 真值 + 有效步数），
   供 ``export.py`` 落盘。
"""
import tqdm
import torch
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
from sklearn import metrics
import logging

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    logger.info('\nauc: ' + str(auc) + 'acc: ' + str(acc))
    print('auc: ' + str(auc) + ' acc: ' + str(acc))
    return auc, acc


def aligned_predictions(p_c, p_t, p_enm, batch, student, q, max_step, index):
    """取单个学生在「预测下一步」上的三路概率与真值，并裁掉 padding。

    原实现（`lossFunc.forward` 内联）逐字搬迁，仅把循环内重复构造的 ``index``
    提为入参。

    Returns:
        (pc, pt, penm, a, valid_len)
        pc/pt/penm 为长度 valid_len 的一维张量（CAHG / DTG / ensemble 三路预测），
        a 为对应的真实作答标签，valid_len 为真实可评测步数。
    """
    delta = batch[student][:, :q] + batch[student][:, q:]

    a = (((batch[student][:, 0:q] -
           batch[student][:, q:]).sum(1) + 1) //
         2)[1:]  # [max_step-1]
    temp_c = p_c[student][:max_step - 1].mm(delta[1:].t())
    temp_t = p_t[student][:max_step - 1].mm(delta[1:].t())
    temp_enm = p_enm[student][:max_step - 1].mm(delta[1:].t())
    pc = temp_c.gather(0, index)[0]
    pt = temp_t.gather(0, index)[0]
    penm = temp_enm.gather(0, index)[0]

    valid_len = len(pc)
    for i in range(len(pc) - 1, -1, -1):
        if pc[i] > 0:
            valid_len = i + 1
            break
    if valid_len < len(pc):
        pc = pc[:valid_len]
        pt = pt[:valid_len]
        penm = penm[:valid_len]
        a = a[:valid_len]

    return pc, pt, penm, a, valid_len


class lossFunc(nn.Module):
    def __init__(self, hidden, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.q = C.NUM_OF_QUESTIONS
        self.hidden = hidden
        self.sig = nn.Sigmoid()
        self.max_step = max_step
        self.device = device
        self.mse = nn.MSELoss()
        # 仅在评估阶段打开，避免训练时产生额外显存/时间开销
        self.collect = False
        self.last_details = None

    def forward(self, logit_c, logit_t, logit_ensemble, batch):
        p_c = self.sig(logit_c)
        p_t = self.sig(logit_t)
        p_enm = self.sig(logit_ensemble)
        '''kd_loss'''
        T = 0.5
        p0_c = self.sig(logit_c/T)
        p0_t = self.sig(logit_t/T)
        p0_enm = self.sig(logit_ensemble/T)
        loss_kd = C.kd_loss * (torch.sum(torch.abs(p0_enm-p0_c)) + torch.sum(torch.abs(p0_enm-p0_t)))
        loss = torch.Tensor([0.0]).cuda()
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        details = [] if self.collect else None

        index = torch.tensor([[i for i in range(self.max_step - 1)]],
                             dtype=torch.long, device=self.device)
        for student in range(batch.shape[0]):
            pc, pt, penm, a, valid_len = aligned_predictions(
                p_c, p_t, p_enm, batch, student, self.q, self.max_step, index)

            loss = loss + self.crossEntropy(pt, a) + self.crossEntropy(pc, a)+ self.crossEntropy(penm, a)
            p_mean = (pt + pc + penm)/3.0
            prediction = torch.cat([prediction, p_mean])
            ground_truth = torch.cat([ground_truth, a])

            if self.collect:
                details.append({
                    'pc': pc.detach().cpu(),
                    'pt': pt.detach().cpu(),
                    'penm': penm.detach().cpu(),
                    'a': a.detach().cpu(),
                    'valid_len': valid_len,
                })

        self.last_details = details
        return loss, loss_kd, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func):
    model.train()
    total = 0.0
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        logit_c, logit_t, logit_ensemble = model(batch)
        loss, loss_kd, _, _ = loss_func(logit_c, logit_t, logit_ensemble, batch)
        total_loss = loss + loss_kd
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        total += total_loss.item()

    return model, optimizer, total / max(len(trainLoader), 1)


def test_epoch(model, testLoader, loss_func, device, collect=False):
    """在给定 loader 上评估。

    Args:
        collect: 为 True 时额外返回逐样本明细列表（含全局样本索引）。
    """
    model.eval()
    loss_func.collect = collect

    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    total_loss = 0.0
    total_kd = 0.0
    details = [] if collect else None
    cursor = 0

    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        with torch.no_grad():
            logit_c, logit_t, logit_ensemble = model(batch)
            loss, loss_kd, p, a = loss_func(logit_c, logit_t, logit_ensemble, batch)
        total_loss += loss.item()
        total_kd += loss_kd.item()
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])

        if collect:
            for local, item in enumerate(loss_func.last_details):
                item['sample_idx'] = cursor + local
                details.append(item)
        cursor += batch.shape[0]

    loss_func.collect = False

    n_batch = max(len(testLoader), 1)
    print('loss:', total_loss / n_batch, '   kd_loss:', total_kd / n_batch)
    print('Total loss:', (total_loss + total_kd) / n_batch)
    auc, acc = performance(ground_truth, prediction)

    if collect:
        return auc, acc, details
    return auc, acc
