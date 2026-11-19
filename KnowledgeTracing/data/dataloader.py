# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : dataloader.py
# @Project: GOODKT
# @Comment :
"""数据加载。

与原版差异：
1. 数据路径统一从 ``Constants`` 的绝对路径常量取，不再依赖 cwd；
2. 新增独立验证集 loader（block 级 90/10 切分产物）；
3. 验证 / 测试集 ``drop_last=False``——评估不应丢弃最后不足一个 batch 的样本；
   训练集保持 ``drop_last=True``（避免 BatchNorm/时序统计被极小 batch 干扰，
   同时与原实现保持一致）。
"""
import os
import sys

sys.path.append('../')
import torch.utils.data as Data
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.preprocess import DataReader
from KnowledgeTracing.data.OneHot import OneHot


def _build_loader(path, max_step, batch_size, shuffle, drop_last, desc):
    handle = DataReader(path, max_step)
    loader = getattr(handle, desc)()
    dataset = OneHot(*loader)
    return Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                           drop_last=drop_last)


def getTrainLoader(train_data_path=None):
    return _build_loader(train_data_path or C.TRAIN_PATH, C.MAX_STEP,
                         C.BATCH_SIZE, shuffle=True, drop_last=True,
                         desc='getTrainData')


def getValidLoader(valid_data_path=None):
    """独立验证集：不参与训练，也不参与转移图构建。"""
    path = valid_data_path or C.VALID_PATH
    if not os.path.exists(path):
        return None
    return _build_loader(path, C.MAX_STEP, C.BATCH_SIZE,
                         shuffle=False, drop_last=False, desc='getTestData')


def getTestLoader(test_data_path=None):
    return _build_loader(test_data_path or C.TEST_PATH, C.MAX_STEP,
                         C.BATCH_SIZE, shuffle=False, drop_last=False,
                         desc='getTestData')


def getLoader(dataset, train_path=None, valid_path=None, test_path=None,
              use_valid=True):
    """返回 ``(trainLoader, validLoader, testLoader)``；``use_valid=False`` 时 valid 为 None。"""
    if dataset not in C.datasets:
        raise ValueError(f'未知数据集: {dataset}')

    if dataset == 'assist2017':
        train_path = train_path or C.TRAIN_PATH
        valid_path = valid_path or C.VALID_PATH
        test_path = test_path or C.TEST_PATH
    else:
        base = os.path.join(C.Dpath, dataset)
        train_path = train_path or os.path.join(base, f'{dataset}_pid_train.csv')
        valid_path = valid_path or os.path.join(base, f'{dataset}_pid_valid.csv')
        test_path = test_path or os.path.join(base, f'{dataset}_pid_test.csv')

    return (getTrainLoader(train_path),
            getValidLoader(valid_path) if use_valid else None,
            getTestLoader(test_path))
