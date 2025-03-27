import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.nn.functional import one_hot

class ImageData():
    def __init__(self, dataset_name):
        # 定义通用的数据预处理（转换为 Tensor 并归一化到 [0,1]）
        transform = transforms.Compose([
            transforms.ToTensor(),  # 输出 Tensor，形状为 (C, H, W)，数值范围 [0,1]
        ])

        if dataset_name == 'mnist':
            dataset_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            dataset_val   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            # 将灰度图转换为 (H, W, 1)
            self.x_train, self.y_train = self._convert_dataset(dataset_train, gray=True)
            self.x_val, self.y_val     = self._convert_dataset(dataset_val, gray=True)
        elif dataset_name == 'cifar10':
            dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            dataset_val   = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            self.x_train, self.y_train = self._convert_dataset(dataset_train, gray=False)
            self.x_val, self.y_val     = self._convert_dataset(dataset_val, gray=False)
        elif dataset_name == 'cifar100':
            dataset_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            dataset_val   = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
            self.x_train, self.y_train = self._convert_dataset(dataset_train, gray=False)
            self.x_val, self.y_val     = self._convert_dataset(dataset_val, gray=False)
        else:
            raise ValueError("Unsupported dataset: {}".format(dataset_name))

        # 此处与原代码一致，直接减去均值（原代码中 x_train_mean 为全 0，可按需求设置）
        x_train_mean = np.zeros(self.x_train.shape[1:], dtype=np.float32)
        self.x_train = self.x_train - x_train_mean
        self.x_val = self.x_val - x_train_mean
        self.x_train_mean = x_train_mean

        self.clip_min = 0.0
        self.clip_max = 1.0

        print('self.x_train.shape:', self.x_train.shape)
        print('self.x_val.shape:', self.x_val.shape)

    def _convert_dataset(self, dataset, gray=False):
        """
        将 torchvision 数据集转换为 numpy 数组
        返回：
          x: (N, H, W, C)，float32
          y: one-hot 编码后的标签，float32
        """
        xs = []
        ys = []
        for img, label in dataset:
            # img 为 Tensor，形状 (C, H, W)
            img_np = img.numpy()  # (C, H, W)
            # 转换为 (H, W, C)
            img_np = np.transpose(img_np, (1, 2, 0))
            if gray and img_np.shape[-1] != 1:
                # 如果需要灰度图，则取第一个通道
                img_np = img_np[..., :1]
            xs.append(img_np)
            ys.append(label)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.int64)
        # 将标签转换为 one-hot 编码（假设类别数为 max(ys)+1）
        num_classes = int(np.max(ys)) + 1
        ys = one_hot(torch.from_numpy(ys), num_classes=num_classes).numpy().astype(np.float32)
        return xs, ys


def split_data(x, y, model, num_classes=10, split_rate=0.8, sample_per_class=100):
    """
    根据模型预测结果过滤出预测正确的样本，
    再按每个类别截取一定数量样本，并按 split_rate 划分训练和测试数据
    """
    np.random.seed(10086)
    # model.predict 接口需返回 numpy 数组，形状为 (N, num_classes)
    pred = model.predict(x)
    label_pred = np.argmax(pred, axis=1)
    label_truth = np.argmax(y, axis=1)
    correct_idx = label_pred == label_truth
    print('Accuracy is {}'.format(np.mean(correct_idx)))
    x, y = x[correct_idx], y[correct_idx]
    label_pred = label_pred[correct_idx]

    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []
    for class_id in range(num_classes):
        idx = np.where(label_pred == class_id)[0]
        # 取每个类别中的前 sample_per_class 个样本
        _x = x[idx][:sample_per_class]
        _y = y[idx][:sample_per_class]
        l = len(_x)
        split_index = int(l * split_rate)
        x_train_list.append(_x[:split_index])
        x_test_list.append(_x[split_index:])
        y_train_list.append(_y[:split_index])
        y_test_list.append(_y[split_index:])

    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    idx_train = np.random.permutation(len(x_train))
    idx_test = np.random.permutation(len(x_test))
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    import argparse
    from build_model import ImageModel  # 请确保该模块已转换为 PyTorch 版本
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['cifar10'], default='cifar10')
    parser.add_argument('--model_name', type=str, choices=['resnet'], default='resnet')
    args = parser.parse_args()

    data_model = args.dataset_name + args.model_name
    dataset = ImageData(args.dataset_name)

    # 构造模型（train=False 且 load=True 表示加载预训练权重）
    model = ImageModel(args.model_name, args.dataset_name, train=False, load=True)

    # 这里以验证集作为待划分数据
    x, y = dataset.x_val, dataset.y_val
    x_train, y_train, x_test, y_test = split_data(x, y, model, num_classes=10, split_rate=0.8)

    print('Split data:')
    print('x_train.shape:', x_train.shape)
    print('x_test.shape:', x_test.shape)
























