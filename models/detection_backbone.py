import torch
from torch import nn

from spikingjelly.activation_based import layer, neuron

from models.MS_ResNet import MSResNetDownBlock
from models.EMS_ResNet import EMSResNetDownBlock
from models.MDS_ResNet import MDSResNetDownBlock
from models.SEW_ResNet import SEWResNetDownBlock
from models.model_utils import MeanDecodeNode
from models.spiking_densenet import DenseNetDownBlock
from models.utils import get_model
from models.SSD_utils import init_weights


class DetectionBackbone(nn.Module):
    def __init__(self, args, step_mode='m'):
        super().__init__()

        self.nz, self.numel = {}, {}
        self.fusion = args.fusion
        self.model = get_model(args)
        self.T = args.T

        if args.pretrained_backbone is not None:
            ckpt = torch.load(args.pretrained_backbone)

            state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
            self.model.load_state_dict(state_dict, strict=False)

        self.out_channels = self.model.out_channels
        extras_fm = args.extras

        BackboneDownBlock = None
        family, version = args.model.split('-')
        if family == 'emsresnet':
            BackboneDownBlock = EMSResNetDownBlock
        elif family == 'mdsresnet':
            BackboneDownBlock = MDSResNetDownBlock
        elif family == 'msresnet':
            BackboneDownBlock = MSResNetDownBlock
        elif family == 'sewresnet':
            BackboneDownBlock = SEWResNetDownBlock
        elif family == 'densenet':
            BackboneDownBlock = DenseNetDownBlock

        self.extras = nn.ModuleList(
            [
                nn.Sequential(
                    BackboneDownBlock(self.out_channels[-1], extras_fm[0],
                                      attention=args.attention) if family == 'mdsresnet' else BackboneDownBlock(
                        self.out_channels[-1], extras_fm[0])
                ),

                nn.Sequential(
                    BackboneDownBlock(extras_fm[0], extras_fm[1],
                                      attention=args.attention) if family == 'mdsresnet' else BackboneDownBlock(
                        extras_fm[0], extras_fm[1]),
                ),
            ]
        )

        if args.decode == 'spiking':
            self.out_module = nn.ModuleList([
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=args.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=args.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=args.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=args.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=args.T),
                ),
            ])

        self.out_channels.extend(extras_fm)

        if self.extras is not None:
            self.extras.apply(init_weights)

    def forward(self, x):
        feature_maps = self.model(x)
        x = feature_maps[-1]
        T = x.shape[0]
        if self.fusion:
            detection_feed = [fm for fm in feature_maps]
        else:
            detection_feed = [self.out_module[i](fm) for i, fm in enumerate(feature_maps)]  # [N, C, H, W]

        if self.extras is not None:
            for i, block in enumerate(self.extras):
                x = block(x)
                if self.fusion:
                    detection_feed.append(x)  # [T, N, C, H, W]
                else:
                    detection_feed.append(self.out_module[i + 3](x))  # [N, C, H, W]

        return detection_feed

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
