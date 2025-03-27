import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
import math
import os

# --------------------------
# ResNet v2 (Bottleneck) 模型定义
# --------------------------
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        # ResNet v2 在每个 block 内采用 BN-ReLU-Conv 顺序
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = F.relu(out)
        # 若需要下采样，则在第一层进行
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class ResNetV2(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetV2, self).__init__()
        self.in_planes = 16

        # CIFAR10 的输入尺寸为 32x32，最初 conv 层不下采样
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 这里不加 BN+ReLU 在最开始，ResNet v2 在前面加了一个预处理层 BN-ReLU
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # 三个 stage，输出通道依次扩展
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.bn_last = nn.BatchNorm2d(self.in_planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_planes, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        # 第一层 block 如果步长不为1或者输入通道不等于输出通道时需要下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入 x: (N, 3, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_last(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

def resnet_v2_cifar(depth, num_classes=10):
    # depth 应满足: depth = 9n + 2, n 为整数
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (e.g., 56, 110)")
    n = (depth - 2) // 9
    layers = [n, n, n]
    model = ResNetV2(BottleneckBlock, layers, num_classes=num_classes)
    return model

# --------------------------
# 学习率调度函数（仿照 Keras 版本）
# --------------------------
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate:", lr)
    return lr

# --------------------------
# 数据增强与数据加载
# --------------------------
def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # CIFAR10 数据集已归一化到 [0,1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

# --------------------------
# 训练与验证
# --------------------------
def train(model, device, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, device, testloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# --------------------------
# 主函数
# --------------------------
def main():
    # 超参数设置
    depth = 110  # 例如：110, 确保满足 9n+2
    num_classes = 10
    batch_size = 128
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = resnet_v2_cifar(depth, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 学习率调度（使用 LambdaLR，每个 epoch 更新）
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch)/1e-3)

    trainloader, testloader = get_dataloaders(batch_size)

    best_acc = 0.0
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, trainloader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, testloader, criterion)
        scheduler.step()

        print("Epoch [{}/{}] Train Loss: {:.4f}  Train Acc: {:.4f}  Val Loss: {:.4f}  Val Acc: {:.4f}".format(
            epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "resnet_v2_best.pth"))

if __name__ == "__main__":
    main()
