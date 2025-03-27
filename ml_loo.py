from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import os
import math
import time
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib
import matplotlib.pyplot as plt

# -------------------- 统计指标函数 --------------------
def con(score):
    # score (n, d)
    score = score.reshape(len(score), -1)
    score_mean = np.mean(score, -1, keepdims=True)
    c_score = np.abs(score - score_mean)
    return np.mean(c_score, axis=-1)

def mad(score):
    pd_list = []
    for i in range(len(score)):
        d = score[i]
        median = np.median(d)
        abs_dev = np.abs(d - median)
        med_abs_dev = np.median(abs_dev)
        pd_list.append(med_abs_dev)
    pd_array = np.array(pd_list)
    return pd_array

def med_pdist(score):
    pd_list = []
    for i in range(len(score)):
        d = score[i]
        k = np.median(pdist(d.reshape(-1, 1)))
        pd_list.append(k)
    return np.array(pd_list)

def pd(score):
    pd_list = []
    for i in range(len(score)):
        d = score[i]
        k = np.mean(pdist(d.reshape(-1, 1)))
        pd_list.append(k)
    return np.array(pd_list)

def neg_kurtosis(score):
    k_list = []
    for i in range(len(score)):
        di = score[i]
        ki = kurtosis(di, nan_policy='raise')
        k_list.append(ki)
    return -np.array(k_list)

def quantile(score):
    # score (n, d)
    score = score.reshape(len(score), -1)
    score_75 = np.percentile(score, 75, axis=-1)
    score_25 = np.percentile(score, 25, axis=-1)
    score_qt = score_75 - score_25
    return score_qt

def calculate(score, stat_name):
    if stat_name == 'variance':
        results = np.var(score, axis=-1)
    elif stat_name == 'std':
        results = np.std(score, axis=-1)
    elif stat_name == 'pdist':
        results = pd(score)
    elif stat_name == 'con':
        results = con(score)
    elif stat_name == 'med_pdist':
        results = med_pdist(score)
    elif stat_name == 'kurtosis':
        results = neg_kurtosis(score)
    elif stat_name == 'skewness':
        results = -skew(score, axis=-1)
    elif stat_name == 'quantile':
        results = quantile(score)
    elif stat_name == 'mad':
        results = mad(score)
    print('results.shape', results.shape)
    return results

# -------------------- 特征提取相关函数 --------------------
def collect_layers(model, interested_layers):
    """
    对于 PyTorch 模型，利用 forward hook 收集感兴趣层的输出。
    本函数会在模型中对所有子模块（除最外层）进行遍历，
    对索引在 interested_layers 中的模块注册 hook，hook 会在前向传播时保存输出：
      - 如果输出为4D（如卷积层特征图），取空间维度均值，输出形状为 (batch, d)
      - 否则直接输出
    返回一个列表 outputs（在 forward 过程中填充）和 hooks 列表（方便后续移除）。
    """
    outputs = []  # 保存各 hook 收集到的输出
    hooks = []

    # 获取所有子模块（按出现顺序）
    modules = list(model.modules())[1:]  # 忽略最外层
    for i, module in enumerate(modules):
        if i in interested_layers:
            def hook(module, input, output, out_list=outputs):
                if isinstance(output, torch.Tensor):
                    if output.dim() == 4:
                        # 平均池化空间维度
                        out_list.append(torch.mean(output, dim=(2, 3)))
                    else:
                        out_list.append(output)
                else:
                    # 若 output 为元组，取第一个元素
                    out = output[0]
                    if out.dim() == 4:
                        out_list.append(torch.mean(out, dim=(2, 3)))
                    else:
                        out_list.append(out)
            h = module.register_forward_hook(hook)
            hooks.append(h)
    return outputs, hooks

def evaluate_features(x, model, interested_layers):
    """
    对输入 x（numpy 数组）分批前向传播，通过注册的 hook 收集中间层输出，
    最后对每个 batch 将各层输出按列拼接，汇总得到所有样本的特征。
    同时，截取最后 num_classes 列作为分类概率，选择最后一条样本的预测标签，
    并将对应列附加到特征上。
    """
    # 若 x 为 (N, H, W, C)，转换为 (N, C, H, W)
    if len(x.shape) == 3:
        _x = np.expand_dims(x, 0)
    else:
        _x = x

    batch_size = 500
    num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))
    collected_features = None

    model.eval()
    with torch.no_grad():
        for i in range(num_iters):
            x_batch = _x[i * batch_size: (i + 1) * batch_size]
            # 转换为 torch tensor，假设输入通道在最后
            x_batch_tensor = torch.from_numpy(x_batch).permute(0, 3, 1, 2).float()
            # 注册 hook，清空 outputs 列表
            outputs, hooks = collect_layers(model, interested_layers)
            outputs.clear()  # 确保为空
            _ = model(x_batch_tensor)
            # 将 hook 收集的输出列表拼接成特征（每个输出 shape 为 (batch, d)）
            batch_features = torch.cat(outputs, dim=1).cpu().numpy()
            if collected_features is None:
                collected_features = batch_features
            else:
                collected_features = np.concatenate([collected_features, batch_features], axis=0)
            # 移除 hook
            for h in hooks:
                h.remove()

    # 假设最后 model.num_classes 列为分类概率（这一部分需要与模型输出对应）
    prob = collected_features[:, -model.num_classes:]
    label = np.argmax(prob[-1])
    # 将对应预测概率作为额外一列附加到特征上
    collected_features = np.concatenate([collected_features, np.expand_dims(prob[:, label], axis=1)], axis=1)
    print('outputs', collected_features.shape)
    return collected_features

def loo_ml_instance(sample, reference, model, interested_layers):
    """
    对单个样本计算 ML-LOO 特征。
    对于输入 sample（形状 (H,W,C)）和参考图 reference，
    生成所有单像素置换的组合，依次前向传播提取特征。
    """
    h, w, c = sample.shape
    sample_flat = sample.reshape(-1)
    reference_flat = reference.reshape(-1)
    # 构造一个 (h*w*c+1, h*w*c) 的布尔矩阵，每行对应一次置换
    positions = np.ones((h * w * c + 1, h * w * c), dtype=bool)
    for i in range(h * w * c):
        positions[i, i] = False
    # 对于每个位置，若位置为 True 则取 sample 中的值，否则取 reference
    data = np.where(positions, sample_flat, reference_flat)
    data = data.reshape((-1, h, w, c))
    features_val = evaluate_features(data, model, interested_layers)  # 形状类似 (num_pixels+1, feature_dim)
    return features_val

def generate_ml_loo_features(args, data_model, reference, model, x, interested_layers):
    """
    对数据 x（字典，包含 'train' 和 'test' 两部分，各有 'original' 和 'adv' 样本）：
      对每个样本利用 ML-LOO 方法计算特征，随后计算各统计量作为最终特征表示，
      并保存到 .npy 文件中。
    """
    # 提取感兴趣层的输出（通过 forward hook 方式在 evaluate_features 中实现）
    cat = {'original': 'ori', 'adv': 'adv', 'noisy': 'noisy'}
    dt = {'train': 'train', 'test': 'test'}
    stat_names = ['std', 'variance', 'con', 'kurtosis', 'skewness', 'quantile', 'mad']

    combined_features = {data_type: {} for data_type in ['test', 'train']}
    for data_type in ['test', 'train']:
        print('data_type', data_type)
        for category in ['original', 'adv']:
            print('category', category)
            all_features = []
            for i, sample in enumerate(x[data_type][category]):
                print('Generating ML-LOO for {}th sample...'.format(i))
                features_val = loo_ml_instance(sample, reference, model, interested_layers)
                # features_val 形状类似 (num_pixels+1, feature_dim)
                print('features_val.shape', features_val.shape)
                # 转置后去掉最后一列（对应附加的概率）
                features_val = np.transpose(features_val)[:, :-1]
                print('features_val.shape', features_val.shape)
                single_feature = []
                for stat_name in stat_names:
                    print('stat_name', stat_name)
                    single_feature.append(calculate(features_val, stat_name))
                single_feature = np.array(single_feature)
                print('single_feature', single_feature.shape)
                all_features.append(single_feature)
            print('all_features', np.array(all_features).shape)
            combined_features[data_type][category] = np.array(all_features)
            save_path = '{}/data/{}_{}_{}_{}_{}.npy'.format(
                data_model,
                args.data_sample,
                dt[data_type],
                cat[category],
                args.attack,
                args.det)
            np.save(save_path, combined_features[data_type][category])
    return combined_features

# -------------------- 其它辅助函数 --------------------
def compute_stat_single_layer(output):
    # 计算单层特征的方差与“集中度”
    variance = np.sum(np.var(output, axis=0))
    con_val = np.sum(np.linalg.norm(output - np.mean(output, axis=0), ord=1, axis=0))
    return variance, con_val

def load_features(data_model, attacks):
    def softmax(x, axis):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)
    cat = {'original': '', 'adv': '_adv', 'noisy': '_noisy'}
    dt = {'train': '_train', 'test': ''}
    features = {attack: {'train': {}, 'test': {}} for attack in attacks}
    normalizer = {}
    for attack in attacks:
        for data_type in ['train', 'test']:
            for category in ['original', 'adv']:
                print('Loading data...')
                file_path = '{}/data/{}{}{}_{}_{}.npy'.format(
                    data_model, 'x_val200',
                    dt[data_type],
                    cat[category],
                    attack,
                    'ml_loo')
                feature = np.load(file_path)
                n = len(feature)
                print('Processing...')
                nums = [0, 64, 64, 128, 128, 256, 256, 10]
                splits = np.cumsum(nums)
                processed = []
                for j, s in enumerate(splits):
                    if j < len(splits) - 1:
                        separated = feature[:, :-1, s:splits[j+1]]
                        if j == len(splits) - 2:
                            separated = softmax(separated, axis=-1)
                        dist = np.var(separated, axis=1)
                        if data_type == 'train' and category == 'original' and attack == 'linfpgd':
                            avg_dist = np.mean(dist, axis=0)
                            normalizer[j] = avg_dist
                        dist = np.sqrt(dist)
                        print(np.mean(dist))
                        processed.append(dist.T)
                processed = np.concatenate(processed, axis=0).T
                print(processed.shape)
                features[attack][data_type][category] = processed
    return features
