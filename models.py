
from torch import nn
from torch.nn import functional as F
import math
import torch

# something for discrete outputs
class SoftmaxDiscreteLinearModel(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(SoftmaxDiscreteLinearModel, self).__init__()
        # set info
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.output_linear = nn.Linear(num_inputs, num_actions)
        self.model_type = 'SoftmaxDiscreteLinearModel'
        if num_actions == 1:
            self.sm = torch.nn.Sigmoid()
        else:
            self.sm = nn.Softmax(dim=1)
    def sample(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return dist.sample(), dist.log_prob(action), torch.argmax(logits,dim=-1)
    def log_prob(self, state, action):
        if not torch.is_tensor(state):
            state = torch.tensor(state).to(self.device)
            action = torch.tensor(action).to(self.device)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)
    def transform_state(self, state):
        return state
    def forward(self, state):
        state = self.transform_state(state)
        logits = self.output_linear(state)
        # logits = torch.clamp(logits, min=-8,max=2)
        probs = self.sm(logits)
        if self.num_actions == 1:
            probs = probs.reshape(-1) 
        return probs

# something for discrete outputs
class DiscreteLinearModel(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DiscreteLinearModel, self).__init__()
        # set info
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.output_linear = nn.Linear(num_inputs, num_actions)
        self.model_type = 'DiscreteLinearModel'
    def sample(self, state):
        logits = self.forward(state)
        logits = torch.clamp(logits, min=-20,max=2)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return dist.sample(), dist.log_prob(action), torch.argmax(logits,dim=-1)
    def log_prob(self, state, action):
        if not torch.is_tensor(state):
            state = torch.tensor(state).to(self.device)
            action = torch.tensor(action).to(self.device)
        logits = self.forward(state)
        logits = torch.clamp(logits, min=-20,max=20)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)
    def transform_state(self, state):
        # if torch.is_tensor(state):
        #     state = state.cpu().numpy()
        return state
    def forward(self, state):
        state = self.transform_state(state)
        logits = self.output_linear(state)
        return logits

# =====================================================
# Something for continous outputs
class ContinuousLinearModel(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ContinuousLinearModel, self).__init__()
        # something to compute it all
        self.mean_linear = nn.Linear(num_inputs, num_actions)

    def forward(self, state):
        state = state.float()
        output = self.mean_linear(state)
        return self.mean_linear(state)

# =====================================================
# Network for mfac
class LinearNueralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True):
        super().__init__()
        self._prediction_params = None
        self.input_size = input_size
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True
        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size, bias=bias)
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)
    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = Z
        logits = self.output_layer(out)
        if self.squeeze_output:
            logits = torch.squeeze(logits)
        return logits

# =====================================================
# MLP
class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()

        self.dropout=dropout
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

            if self.dropout:
                out = F.dropout(out, p=0.5)

        logits = self.output_layer(out)

        return logits

# =====================================================
# ResNet
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        block = BasicBlock
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_DenseNet(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(num_classes):
    return DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=32,
        num_classes=num_classes)

def DenseNet169():
    return DenseNet(Bottleneck_DenseNet, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck_DenseNet, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck_DenseNet, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=12)
