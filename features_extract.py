import argparse
import time
import logging
import os
import numpy as np
import torch

from build_model import ImageModel
from load_data import ImageData, split_data
from attack_model import Attack, CW
from ml_loo import generate_ml_loo_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, 
                        choices=['cifar10'], 
                        default='cifar10')
    parser.add_argument('--model_name', type=str, 
                        choices=['resnet'], 
                        default='resnet')
    parser.add_argument('--data_sample', type=str, 
                        choices=['x_train', 'x_val', 'x_val200'], 
                        default='x_val200')
    parser.add_argument('--attack', type=str, 
                        choices=['cw'], 
                        default='cw')
    parser.add_argument('--det', type=str, 
                        choices=['ml_loo'], 
                        default='ml_loo')

    args = parser.parse_args()
    data_model = args.dataset_name + args.model_name

    print('Loading dataset...')
    dataset = ImageData(args.dataset_name)
    # 构造模型，此处 train=False 且 load=True 表示加载预训练权重
    model = ImageModel(args.model_name, args.dataset_name, train=False, load=True)

    ###########################################################
    # 加载原始、对抗和噪声样本
    ###########################################################
    print('Loading original, adversarial and noisy samples...')
    X_test = np.load('{}/data/{}_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))
    X_test_adv = np.load('{}/data/{}_adv_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))
    X_train = np.load('{}/data/{}_train_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))
    X_train_adv = np.load('{}/data/{}_train_adv_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

    # 使用模型进行预测（ImageModel.predict 已实现为 PyTorch 版本）
    Y_test = model.predict(X_test)
    print("X_test_adv: ", X_test_adv.shape)

    # 将数据组织成字典
    x = {
        'train': {
            'original': X_train,
            'adv': X_train_adv,
        },
        'test': {
            'original': X_test,
            'adv': X_test_adv,
        },
    }

    #################################################################
    # 提取原始、对抗和噪声样本的特征
    #################################################################
    cat = {'original': 'ori', 'adv': 'adv', 'noisy': 'noisy'}
    dt = {'train': 'train', 'test': 'test'}

    if args.det in ['ml_loo']:
        # 对于 resnet 模型，选择感兴趣的层索引（注意：该索引应与 ImageModel 内部实现保持一致）
        if args.model_name == 'resnet':
            interested_layers = [14, 24, 35, 45, 56, 67, 70]

        print('extracting layers ', interested_layers)
        # 假设 dataset 中保存了训练数据均值，作为参考图像（例如：dataset.x_train_mean）
        reference = - dataset.x_train_mean

        combined_features = generate_ml_loo_features(args, data_model, reference, model, x, interested_layers)

        for data_type in ['test', 'train']:
            for category in ['original', 'adv']:
                save_path = '{}/data/{}_{}_{}_{}_{}.npy'.format(
                    data_model,
                    args.data_sample,
                    dt[data_type],
                    cat[category],
                    args.attack,
                    args.det)
                np.save(save_path, combined_features[data_type][category])
