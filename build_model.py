import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import math
import time

#############################################
# 网络模型定义
#############################################

# SCNN: 两层卷积+池化，接全连接
class SCNN(nn.Module):
    def __init__(self, input_size=28, channel=1, num_classes=10):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=5, padding=2)  # 'same' padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * (input_size // 4) * (input_size // 4), 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 降采样
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        output = F.softmax(logits, dim=1)
        return output, logits  # 返回 softmax 后的概率和 logits

# CNN: 类似 Keras 版本的较深卷积网络
class CNN(nn.Module):
    def __init__(self, input_size=32, channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(channel, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3)  # 无 padding
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3)  # 无 padding
        self.conv5 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3)  # 无 padding
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.dropout_fc2 = nn.Dropout(0.5)
        
        # 根据卷积和池化计算展平后的尺寸，这里采用经验值，可根据实际输入调整
        self.fc1 = nn.Linear(192 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        logits = self.fc3(x)
        output = F.softmax(logits, dim=1)
        return output, logits

# FC: 全连接网络
class FC(nn.Module):
    def __init__(self, input_size=28, channel=1, num_classes=10):
        super(FC, self).__init__()
        self.input_dim = input_size * input_size * channel
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        output = F.softmax(logits, dim=1)
        return output, logits

#############################################
# 网络构造函数（可根据 dataset_name 和 model_name 选择模型）
#############################################
def construct_original_network(dataset_name, model_name, train=True):
    if dataset_name.lower() == 'mnist':
        input_size = 28
        num_classes = 10
        channel = 1
    elif dataset_name.lower() in ['cifar10', 'cifar100']:
        input_size = 32
        num_classes = 10 if dataset_name.lower() == 'cifar10' else 100
        channel = 3
    else:
        raise ValueError("Unsupported dataset")
    
    if model_name.lower() == 'scnn':
        model = SCNN(input_size, channel, num_classes)
    elif model_name.lower() == 'cnn':
        model = CNN(input_size, channel, num_classes)
    elif model_name.lower() == 'fc':
        model = FC(input_size, channel, num_classes)
    # 对于 resnet 和 densenet 可考虑调用 torchvision.models 进行加载
    else:
        raise ValueError("Unsupported model")
    
    return model, input_size, num_classes

#############################################
# ImageModel 类（封装模型训练、对抗训练、预测、梯度计算等功能）
#############################################
class ImageModel:
    def __init__(self, model_name, dataset_name, train=False, load_path=None, device=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.framework = 'pytorch'
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print('Constructing network...')
        self.model, self.input_size, self.num_classes = construct_original_network(dataset_name, model_name, train=train)
        self.model = self.model.to(self.device)
        
        if load_path is not None:
            print("Loading model weights from {}".format(load_path))
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        
        self.criterion = nn.CrossEntropyLoss()
        # 优化器等在 train/adv_train 中构造
        
        self.pred_counter = 0

    def train(self, train_dataset, val_dataset, epochs=20, batch_size=128, lr=0.001):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                # 输出：(_, logits)
                _, logits = self.model(data)
                loss = self.criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            print("Epoch [{}/{}] Train Loss: {:.4f}  Train Acc: {:.4f}".format(epoch+1, epochs, train_loss, train_acc))
            
            # 验证
            self.model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    _, logits = self.model(data)
                    pred = logits.argmax(dim=1)
                    correct_val += pred.eq(target).sum().item()
                    total_val += data.size(0)
            val_acc = correct_val / total_val
            print("Validation Acc: {:.4f}".format(val_acc))
    
    def adv_train(self, train_dataset, val_dataset, attack_name='fgsm', epochs=20, batch_size=128, lr=0.001, eps=0.3):
        """
        对抗训练示例，支持 FGSM 和 PGD（Linf）攻击。
        这里使用简单的实现，您可以根据需要扩展。
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        def fgsm_attack(data, target, eps):
            data.requires_grad = True
            self.model.eval()
            _, logits = self.model(data)
            loss = self.criterion(logits, target)
            self.model.zero_grad()
            loss.backward()
            # FGSM perturbation
            data_grad = data.grad.data
            perturbed_data = data + eps * data_grad.sign()
            # clip to [0,1]
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            return perturbed_data
        
        def pgd_attack(data, target, eps, alpha=0.01, iters=10):
            original_data = data.clone().detach()
            perturbed_data = data.clone().detach()
            for i in range(iters):
                perturbed_data.requires_grad = True
                self.model.eval()
                _, logits = self.model(perturbed_data)
                loss = self.criterion(logits, target)
                self.model.zero_grad()
                loss.backward()
                grad = perturbed_data.grad.data
                perturbed_data = perturbed_data + alpha * grad.sign()
                # projection onto eps-ball
                eta = torch.clamp(perturbed_data - original_data, min=-eps, max=eps)
                perturbed_data = torch.clamp(original_data + eta, 0, 1).detach()
            return perturbed_data
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                # 生成对抗样本
                if attack_name.lower() == 'fgsm':
                    data_adv = fgsm_attack(data, target, eps)
                elif attack_name.lower() == 'pgd':
                    data_adv = pgd_attack(data, target, eps)
                else:
                    raise ValueError("Unsupported attack for adv_train")
                
                optimizer.zero_grad()
                # 对抗训练使用对抗样本
                _, logits = self.model(data_adv)
                loss = self.criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            print("Adv Train Epoch [{}/{}] Loss: {:.4f}  Acc: {:.4f}".format(epoch+1, epochs, train_loss, train_acc))
            
            # 验证：分别计算原始和对抗样本上的准确率
            self.model.eval()
            correct_nat = 0
            correct_adv = 0
            total_val = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    # 自然样本预测
                    _, logits_nat = self.model(data)
                    pred_nat = logits_nat.argmax(dim=1)
                    correct_nat += pred_nat.eq(target).sum().item()
                    # 对抗样本预测（使用 FGSM 生成对抗样本作为示例）\n                    data_adv = fgsm_attack(data, target, eps)  \n                    _, logits_adv = self.model(data_adv)\n                    pred_adv = logits_adv.argmax(dim=1)\n                    correct_adv += pred_adv.eq(target).sum().item()\n                    total_val += data.size(0)\n            print("Validation Nat Acc: {:.4f}  Adv Acc: {:.4f}".format(correct_nat/total_val, correct_adv/total_val))
    
    def predict(self, x, batch_size=500, logits=False):
        """
        x: numpy 数组，形状为 (N, H, W) 或 (N, H, W, C)
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            # 根据数据维度做简单判断
            if x.ndim == 3:
                # (N, H, W) -> (N, 1, H, W)
                x = np.expand_dims(x, axis=1)
            elif x.ndim == 4 and x.shape[-1] in [1,3]:
                # Keras 默认通道在最后，转换为 PyTorch 格式 (N, C, H, W)
                x = np.transpose(x, (0, 3, 1, 2))
            x_tensor = torch.from_numpy(x).float().to(self.device)
        else:
            x_tensor = x.to(self.device)
        
        preds = []
        with torch.no_grad():
            loader = DataLoader(x_tensor, batch_size=batch_size)
            for batch in loader:
                out, out_logits = self.model(batch)
                if logits:
                    preds.append(out_logits.cpu().numpy())
                else:
                    preds.append(out.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        self.pred_counter += preds.shape[0]
        return preds
    
    def compute_saliency(self, x, saliency_type='gradient'):
        """
        计算输入 x 的 saliency 图，这里仅支持 gradient 方式。
        x: numpy 数组，形状 (N, H, W) 或 (N, H, W, C)
        返回与 x 相同 shape 的梯度绝对值。
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = np.expand_dims(x, axis=1)
            elif x.ndim == 4 and x.shape[-1] in [1,3]:
                x = np.transpose(x, (0, 3, 1, 2))
            x_tensor = torch.from_numpy(x).float().to(self.device)
        else:
            x_tensor = x.to(self.device)
        
        x_tensor.requires_grad = True
        out, logits = self.model(x_tensor)
        # 取预测类别对应的 logits
        preds = logits.argmax(dim=1)
        loss = self.criterion(logits, preds)
        self.model.zero_grad()
        loss.backward()
        # saliency 为输入梯度的绝对值
        saliency = x_tensor.grad.data.abs().cpu().numpy()
        # 如需要，可转回原来的通道顺序
        if saliency.shape[1] in [1,3]:
            saliency = np.transpose(saliency, (0, 2, 3, 1))
        return saliency

    def compute_ig(self, x, reference, steps=50):
        """
        简单实现 Integrated Gradients 算法
        x: numpy 数组 (N, H, W, C) 或 (N, C, H, W)
        reference: 同 x 尺寸的基准图（例如全0图像）
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            # 如果输入为 (N, H, W, C)，转换为 (N, C, H, W)
            if x.ndim == 4 and x.shape[-1] in [1,3]:
                x_tensor = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float().to(self.device)
            else:
                x_tensor = torch.from_numpy(x).float().to(self.device)
            if reference is None:
                reference = torch.zeros_like(x_tensor)
            else:
                if isinstance(reference, np.ndarray):
                    if reference.ndim == 4 and reference.shape[-1] in [1,3]:
                        reference = torch.from_numpy(np.transpose(reference, (0, 3, 1, 2))).float().to(self.device)
                    else:
                        reference = torch.from_numpy(reference).float().to(self.device)
        else:
            x_tensor = x.to(self.device)
            reference = reference.to(self.device)
        
        # 构造渐进序列
        scaled_inputs = [reference + (float(i) / steps) * (x_tensor - reference) for i in range(1, steps+1)]
        ig = torch.zeros_like(x_tensor)
        
        for scaled in scaled_inputs:
            scaled.requires_grad = True
            out, logits = self.model(scaled)
            preds = logits.argmax(dim=1)
            loss = self.criterion(logits, preds)
            self.model.zero_grad()
            loss.backward()
            ig += scaled.grad.data
        
        # 平均后乘以 (x - reference)
        ig = (x_tensor - reference) * ig / steps
        # 转为 numpy，并恢复通道顺序为 (N, H, W, C)
        ig = ig.cpu().numpy()
        ig = np.transpose(ig, (0, 2, 3, 1))
        return ig


