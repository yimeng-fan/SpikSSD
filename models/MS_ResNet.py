import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer
from .model_utils import batch_norm_2d, batch_norm_2d1

# Model for MS-ResNet
class MSResNetDownBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, k_size=3, stride=2, e=0.5, step_mode='m', backend='cupy'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        c_ = int(out_channels * BasicBlock_18.expansion * e)  # hidden channels
        pad = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=k_size,
                         stride=stride,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(c_,
                         out_channels * BasicBlock_18.expansion,
                         kernel_size=k_size,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels * BasicBlock_18.expansion),
        )

        self.shortcut = nn.Sequential(
            layer.Conv2d(in_channels,
                         out_channels * BasicBlock_18.expansion,
                         kernel_size=1,
                         stride=stride,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(out_channels * BasicBlock_18.expansion),
        )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5, step_mode='m', backend='cupy'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        c_ = int(out_channels * BasicBlock_18.expansion * e)  # hidden channels
        pad = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=k_size,
                         stride=stride,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(c_,
                         out_channels * BasicBlock_18.expansion,
                         kernel_size=k_size,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels * BasicBlock_18.expansion),
        )
        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels,
                             out_channels * BasicBlock_18.expansion,
                             kernel_size=1,
                             stride=stride,
                             bias=False,
                             step_mode=step_mode),
                batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet_origin_18(nn.Module):
    # Channel:
    def __init__(self, block, num_block, step_mode='m', backend='cupy', num_classes=1000):
        super().__init__()
        k = 1
        self.nz, self.numel = {}, {}
        self.in_channels = 64 * k
        self.step_mode = step_mode
        self.backend = backend
        self.out_channels = [128 * k, 256 * k, 512 * k]
        self.conv1 = nn.Sequential(
            layer.Conv2d(4,
                         64 * k,
                         kernel_size=7,
                         padding=3,
                         bias=False,
                         stride=2,
                         step_mode=self.step_mode),
            batch_norm_2d(64 * k),
        )
        self.pool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode=self.step_mode)

        self.mem_update = neuron.LIFNode(step_mode=step_mode, backend=self.backend)
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.classifier = nn.Sequential(
            layer.Conv2d(512 * block.expansion * k, num_classes, kernel_size=1, bias=False,
                         step_mode=self.step_mode),
            batch_norm_2d(num_classes),
            neuron.LIFNode(step_mode=step_mode, backend=self.backend),
        )


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride,
                                step_mode=self.step_mode, backend=self.backend))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, classify=False):
        if classify:
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.mem_update(output)
            output = self.classifier(output)
            output = output.flatten(start_dim=-2).sum(dim=-1)
            output = output.sum(dim=0) / output.size()[0]
            return output
        else:
            output_list = []
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output_list.append(output)

            output = self.conv4_x(output)
            output_list.append(output)

            output = self.conv5_x(output)
            output_list.append(output)

            return output_list

    def add_hooks(self, instance):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()

            return hook

        self.hooks = {}

        for name, module in self.named_modules():
            if isinstance(module, instance):
                self.nz[name], self.numel[name] = 0, 0
                self.hooks[name] = module.register_forward_hook(get_nz(name))

    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0

    def get_nz_numel(self):
        return self.nz, self.numel


def ms_resnet18(num_classes, backend):
    return ResNet_origin_18(BasicBlock_18, [2, 2, 2, 2], num_classes=num_classes, backend=backend)


def ms_resnet34(num_classes, backend):
    return ResNet_origin_18(BasicBlock_18, [3, 4, 6, 3], num_classes=num_classes, backend=backend)
