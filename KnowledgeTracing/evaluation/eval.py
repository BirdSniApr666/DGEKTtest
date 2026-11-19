"""DGEKT losses and evaluation shared by the population and personal graph models."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn import metrics

from KnowledgeTracing.Constant import Constants as C


logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    y = ground_truth.detach().cpu().numpy()
    p = prediction.detach().cpu()
    if not len(y):
        raise ValueError('No valid next-question targets in evaluation data')
    auc = float(metrics.roc_auc_score(y, p.numpy())) if y.min() != y.max() else float('nan')
    acc = float(metrics.accuracy_score(y, p.round().numpy()))
    logger.info('auc: %s acc: %s', auc, acc)
    print(f'auc: {auc} acc: {acc}')
    return auc, acc


def _nodes(batch):
    if isinstance(batch, dict):
        return batch['nodes']
    # Historical loaders yield a one-hot vector of interaction IDs at each step.
    magnitude, nodes = batch.max(dim=-1)
    return nodes.masked_fill(magnitude == 0, -1)


def _next_targets(nodes, question_count):
    target = nodes[:, 1:]
    valid = (nodes[:, :-1] >= 0) & (target >= 0)
    questions = target.clamp_min(0).remainder(question_count).unsqueeze(-1)
    answers = (target < question_count) & valid
    return questions, answers, valid


def _select_next(values, questions):
    # Output at t represents history through t. Query only column q[t+1].
    return values[:, :-1].gather(-1, questions).squeeze(-1)


def aligned_predictions(p_c, p_t, p_enm, batch, student, q, max_step, index):
    """Historical single-chunk API for callers that already have probabilities."""
    nodes = _nodes(batch)[student:student + 1, :max_step]
    questions, answers, valid = _next_targets(nodes, q)
    probabilities = [_select_next(p[student:student + 1, :max_step], questions)[valid]
                     for p in (p_c, p_t, p_enm)]
    labels = answers[valid].to(p_c.dtype)
    return (*probabilities, labels, labels.numel())


class lossFunc(nn.Module):
    def __init__(self, hidden, max_step, device):
        super().__init__()
        self.q = C.NUM_OF_QUESTIONS
        self.device = device
        self.collect = False
        self.last_details = None

    def forward(self, logit_c, logit_t, logit_ensemble, batch):
        logits = (logit_c, logit_t, logit_ensemble)
        questions, answers, valid = _next_targets(_nodes(batch), self.q)
        answers = answers.to(logit_c.dtype)
        pc, pt, pe = [torch.sigmoid(_select_next(value, questions)) for value in logits]

        # Preserve the baseline's sum of per-student mean BCE across three heads.
        # A singleton tail chunk has no supervised targets and contributes zero CE.
        n_valid = valid.sum(dim=1).clamp_min(1)
        loss = sum(((F.binary_cross_entropy(p, answers, reduction='none') * valid).sum(dim=1)
                    / n_valid).sum() for p in (pt, pc, pe))

        # Preserve baseline KD over all questions and timesteps (including padding).
        teacher = torch.sigmoid(logit_ensemble / 0.5)
        loss_kd = C.kd_loss * sum((teacher - torch.sigmoid(peer / 0.5)).abs().sum()
                                  for peer in (logit_c, logit_t))
        probability = (pt + pc + pe) / 3.0
        self.last_details = None
        if self.collect:
            # Transfer once per tensor instead of synchronizing at every student.
            pc_cpu, pt_cpu, pe_cpu, p_cpu, y_cpu, mask_cpu = [
                x.detach().cpu() for x in (pc, pt, pe, probability, answers, valid)
            ]
            self.last_details = []
            for i, mask in enumerate(mask_cpu):
                y = y_cpu[i, mask]
                self.last_details.append({
                    'pc': pc_cpu[i, mask], 'pt': pt_cpu[i, mask], 'penm': pe_cpu[i, mask],
                    'a': y, 'p_mean': p_cpu[i, mask], 'valid_len': y.numel(),
                    'steps': mask.nonzero(as_tuple=True)[0] + 1,
                })

        return loss, loss_kd, probability[valid], answers[valid]


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return batch.to(device, non_blocking=True)


def train_epoch(model, trainLoader, optimizer, loss_func):
    model.train()
    total = 0.0
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = _to_device(batch, loss_func.device)
        optimizer.zero_grad()
        loss, loss_kd, _, _ = loss_func(*model(batch), batch)
        total_loss = loss + loss_kd
        total_loss.backward()
        optimizer.step()
        total += total_loss.item()
    return model, optimizer, total / max(len(trainLoader), 1)


@torch.no_grad()
def test_epoch(model, testLoader, loss_func, device, collect=False):
    """Evaluate a split. Collected sample indices assume the sequential evaluation loader."""
    model.eval()
    loss_func.collect = collect
    predictions, labels = [], []
    total_loss = total_kd = 0.0
    details = []
    try:
        for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
            batch = _to_device(batch, device)
            loss, loss_kd, p, a = loss_func(*model(batch), batch)
            total_loss += loss.item()
            total_kd += loss_kd.item()
            predictions.append(p.cpu())
            labels.append(a.cpu())
            if collect:
                for item in loss_func.last_details:
                    item['sample_idx'] = len(details)
                    details.append(item)
    finally:
        loss_func.collect = False
        loss_func.last_details = None

    if not predictions:
        raise ValueError('Evaluation loader has no chunks')
    n_batch = len(predictions)
    print('loss:', total_loss / n_batch, '   kd_loss:', total_kd / n_batch)
    print('Total loss:', (total_loss + total_kd) / n_batch)
    auc, acc = performance(torch.cat(labels), torch.cat(predictions))
    return (auc, acc, details) if collect else (auc, acc)
