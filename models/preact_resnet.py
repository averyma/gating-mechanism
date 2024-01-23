'''https://raw.githubusercontent.com/kuangliu/pytorch-cifar/master/models/preact_resnet.py'''

'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PreActResNet_s(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_s, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PreActResNet_gate_v1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_gate_v1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        # added for gated mechanism
        sigmoid = nn.Sigmoid()
        d1 = 64 * block.expansion * 32**2
        d2 = 128 * block.expansion * 16**2
        d3 = 256 * block.expansion * 8**2
        d4 = 512 * block.expansion

        self.Gate1 = nn.Sequential(nn.Linear(d1, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))
        self.Gate2 = nn.Sequential(nn.Linear(d2, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))
        self.Gate3 = nn.Sequential(nn.Linear(d3, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))
        self.Gate4 = nn.Sequential(nn.Linear(d4, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))

        self.fc1 = nn.Linear(d1, num_classes)
        self.fc2 = nn.Linear(d2, num_classes)
        self.fc3 = nn.Linear(d3, num_classes)
        self.fc4 = nn.Linear(d4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        gate = torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.layer1(x * gate)
        x_flatten = torch.flatten(x, 1)
        gate1 = self.Gate1(x_flatten)
        gate = gate*(1 - gate1)
        logit1 = self.fc1(x_flatten * (1-gate).flatten(1))

        x = self.layer2(x * gate)
        x_flatten = torch.flatten(x, 1)
        gate2 = self.Gate2(x_flatten)
        gate = gate*(1 - gate2)
        logit2 = self.fc2(x_flatten * (1-gate).flatten(1))

        x = self.layer3(x * gate)
        x_flatten = torch.flatten(x, 1)
        gate3 = self.Gate3(x_flatten)
        gate = gate*(1 - gate3)
        logit3 = self.fc3(x_flatten * (1-gate).flatten(1))

        x = self.layer4(x * gate)

        x = self.avgpool(x)
        x_flatten = torch.flatten(x, 1)
        gate4 = self.Gate4(x_flatten)
        logit4 = self.fc4(x_flatten)

        concatgate = torch.cat([gate1, gate2, gate3, gate4], dim=1).squeeze()
        concatlogit = torch.cat([logit1.unsqueeze(2),
                                 logit2.unsqueeze(2),
                                 logit3.unsqueeze(2),
                                 logit4.unsqueeze(2)], dim=2)
        return concatgate, concatlogit

class PreActResNet_gate_v2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_gate_v2, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        # added for gated mechanism
        sigmoid = nn.Sigmoid()
        d1 = 64 * block.expansion * 32**2
        d2 = 128 * block.expansion * 16**2
        d3 = 256 * block.expansion * 8**2
        d4 = 512 * block.expansion

        self.Gate1 = nn.Sequential(nn.Linear(d1, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))
        self.Gate2 = nn.Sequential(nn.Linear(d2, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))
        self.Gate3 = nn.Sequential(nn.Linear(d3, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))
        self.Gate4 = nn.Sequential(nn.Linear(d4, 1), sigmoid, nn.Unflatten(1, (1, 1, 1)))

        self.fc1 = nn.Linear(d1, num_classes)
        self.fc2 = nn.Linear(d2, num_classes)
        self.fc3 = nn.Linear(d3, num_classes)
        self.fc4 = nn.Linear(d4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        # gate = torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.layer1(x)
        x_flatten = torch.flatten(x, 1)
        gate1 = self.Gate1(x_flatten)
        # gate = gate*(1 - gate1)
        logit1 = self.fc1(x_flatten * (1-gate1).flatten(1))

        x = self.layer2(x * gate1)
        x_flatten = torch.flatten(x, 1)
        gate2 = self.Gate2(x_flatten)
        # gate = gate*(1 - gate2)
        logit2 = self.fc2(x_flatten * (1-gate2).flatten(1))

        x = self.layer3(x * gate2)
        x_flatten = torch.flatten(x, 1)
        gate3 = self.Gate3(x_flatten)
        # gate = gate*(1 - gate3)
        logit3 = self.fc3(x_flatten * (1-gate3).flatten(1))

        x = self.layer4(x * gate3)

        x = self.avgpool(x)
        x_flatten = torch.flatten(x, 1)
        gate4 = self.Gate4(x_flatten)
        logit4 = self.fc4(x_flatten)

        concatgate = torch.cat([gate1, gate2, gate3, gate4], dim=1).squeeze()
        concatlogit = torch.cat([logit1.unsqueeze(2),
                                 logit2.unsqueeze(2),
                                 logit3.unsqueeze(2),
                                 logit4.unsqueeze(2)], dim=2)
        return concatgate, concatlogit

def PreActResNet18(gate, shallow, num_classes=10):
    if shallow:
        return PreActResNet_s(PreActBlock, [2,2,2,2], num_classes)
    elif gate == 1:
        return PreActResNet_gate_v1(PreActBlock, [2,2,2,2], num_classes)
    elif gate == 2:
        return PreActResNet_gate_v2(PreActBlock, [2,2,2,2], num_classes)
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes)

def PreActResNet50(gate, shallow, num_classes=10):
    if shallow:
        return PreActResNet_s(PreActBottleneck, [3,4,6,3], num_classes)
    elif gate == 1:
        return PreActResNet_gate_v1(PreActBottleneck, [3,4,6,3], num_classes)
    elif gate == 2:
        return PreActResNet_gate_v2(PreActBottleneck, [3,4,6,3], num_classes)
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes)

def PreActResNet101(gate, shallow, num_classes=10):
    if shallow:
        return PreActResNet_s(PreActBottleneck, [3,4,23,3], num_classes)
    elif gate == 1:
        return PreActResNet_gate_v1(PreActBottleneck, [3,4,23,3], num_classes)
    elif gate == 2:
        return PreActResNet_gate_v2(PreActBottleneck, [3,4,23,3], num_classes)
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
