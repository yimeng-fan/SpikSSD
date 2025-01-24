from torch import nn, Tensor

from spikingjelly.activation_based import layer, neuron, surrogate, functional

from .model_utils import batch_norm_2d, batch_norm_2d1


class SpikingFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, step_mode='m', backend='cupy', args=None):
        super().__init__()
        c_ = int(out_channels * 1.)  # hidden channels
        self.T = args.T

        self.residual_function = nn.Sequential(
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(c_,
                         out_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
            SpikingSeparableConv(in_channels, out_channels, stride=1, step_mode=step_mode, backend=backend, args=args),
            # neuron.LIFNode(step_mode=step_mode, backend=backend),
            # layer.Conv2d(in_channels,
            #              out_channels,
            #              kernel_size=1,
            #              stride=1,
            #              bias=False,
            #              step_mode=step_mode),
            # batch_norm_2d(out_channels),
        )

    def forward(self, x):
        assert x.shape[0] == self.T
        return self.residual_function(x) + self.shortcut(x)


class SpikingDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=(0, 0), step_mode='m', backend='cupy', args=None):
        super().__init__()
        self.T = args.T

        self.down_function = nn.Sequential(
            layer.MaxPool2d(2, stride=2, padding=padding, step_mode=step_mode),
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )

    def forward(self, x):
        assert x.shape[0] == self.T
        return self.down_function(x)


class SpikingUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_size, step_mode='m', backend='cupy', args=None):
        super().__init__()
        self.T = args.T

        self.up_function = nn.Sequential(
            layer.MultiStepContainer(nn.Upsample(size=out_size, mode='nearest')),
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )

    def forward(self, x):
        assert x.shape[0] == self.T
        return self.up_function(x)


class SpikingSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, step_mode='m', backend='cupy', args=None):
        super().__init__()

        self.T = args.T
        self.separable_conv = nn.Sequential(
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         in_channels,
                         kernel_size=3,
                         stride=stride,
                         padding=1,
                         groups=in_channels,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(in_channels),
            neuron.LIFNode(step_mode=step_mode, backend=backend),
            layer.Conv2d(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(out_channels),
        )

    def forward(self, x):
        assert x.shape[0] == self.T
        return self.separable_conv(x)
