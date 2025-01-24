from models.model_utils import MeanDecodeNode
from models.neck_utils import *


class DetectionNeck(nn.Module):
    def __init__(self, in_channels, channels=128, attention=False, step_mode='m', backend='cupy', args=None):
        super().__init__()

        self.epsilon = 1e-4
        self.nz, self.numel = {}, {}
        s3_c, s4_c, s5_c, s6_c, s7_c = in_channels
        self.out_channels = [channels] * 5
        self.T = args.T
        self.attention = attention

        # Spatial upsampling
        self.s7_s6_up = SpikingUpBlock(s7_c, s6_c, (4, 5), step_mode=step_mode, backend=backend, args=args)
        self.s6_s5_up = SpikingUpBlock(channels, s5_c, (8 , 10), step_mode=step_mode, backend=backend, args=args)
        self.s5_s4_up = SpikingUpBlock(channels, s4_c, (15, 19), step_mode=step_mode, backend=backend, args=args)
        self.s4_s3_up = SpikingUpBlock(channels, s3_c, (30, 38), step_mode=step_mode, backend=backend, args=args)

        # Spatial downsampling
        self.s3_s4_down = SpikingDownBlock(channels, channels, padding=(0, 0), step_mode=step_mode, backend=backend,
                                           args=args)
        self.s4_s5_down = SpikingDownBlock(channels, channels, padding=(1, 1), step_mode=step_mode, backend=backend,
                                           args=args)
        self.s5_s6_down = SpikingDownBlock(channels, channels, padding=(0, 0), step_mode=step_mode, backend=backend,
                                           args=args)
        self.s6_s7_down = SpikingDownBlock(channels, s7_c, padding=(0, 1), step_mode=step_mode, backend=backend,
                                           args=args)

        # Upsampling stage fusion module
        self.s6_up_fusion = SpikingFusionBlock(s6_c, channels, attention=self.attention, step_mode=step_mode,
                                               backend=backend, args=args)
        self.s5_up_fusion = SpikingFusionBlock(s5_c, channels, attention=self.attention, step_mode=step_mode,
                                               backend=backend, args=args)
        self.s4_up_fusion = SpikingFusionBlock(s4_c, channels, attention=self.attention, step_mode=step_mode,
                                               backend=backend, args=args)
        self.s3_up_fusion = SpikingFusionBlock(s3_c, channels, attention=self.attention, step_mode=step_mode,
                                               backend=backend, args=args)

        # Downsampling stage fusion module
        self.s4_down_fusion = SpikingFusionBlock(channels, channels, attention=self.attention, step_mode=step_mode,
                                                 backend=backend, args=args)
        self.s5_down_fusion = SpikingFusionBlock(channels, channels, attention=self.attention, step_mode=step_mode,
                                                 backend=backend, args=args)
        self.s6_down_fusion = SpikingFusionBlock(channels, channels, attention=self.attention, step_mode=step_mode,
                                                 backend=backend, args=args)
        self.s7_down_fusion = SpikingFusionBlock(s7_c, channels, attention=self.attention, step_mode=step_mode,
                                                 backend=backend, args=args)

        # output neuron
        if args.decode == 'spiking':
            self.out_module = nn.ModuleList([
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=self.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=self.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=self.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=self.T),
                ),
                nn.Sequential(
                    neuron.LIFNode(step_mode='m', backend='cupy'),
                    MeanDecodeNode(T=self.T),
                ),
            ])

    def forward(self, source_features):
        assert len(source_features) == 5
        output_features = []

        '''print(f"source_features[0]: {source_features[0].shape}")
        print(f"source_features[1]: {source_features[1].shape}")
        print(f"source_features[2]: {source_features[2].shape}")
        print(f"source_features[3]: {source_features[3].shape}")
        print(f"source_features[4]: {source_features[4].shape}")'''

        T = source_features[0].shape[0]
        assert T == self.T

        # Obtain input features
        s3_in = source_features[0]
        s4_in = source_features[1]
        s5_in = source_features[2]
        s6_in = source_features[3]
        s7_in = source_features[4]

        # Upsampling fusion
        s6_up = self.s6_up_fusion(s6_in + self.s7_s6_up(s7_in))
        s5_up = self.s5_up_fusion(s5_in + self.s6_s5_up(s6_up))
        s4_up = self.s4_up_fusion(s4_in + self.s5_s4_up(s5_up))
        s3_out = self.s3_up_fusion(s3_in + self.s4_s3_up(s4_up))

        # Downsampling fusion
        s4_out = self.s4_down_fusion(s4_up + self.s3_s4_down(s3_out))
        s5_out = self.s5_down_fusion(s5_up + self.s4_s5_down(s4_out))
        s6_out = self.s6_down_fusion(s6_up + self.s5_s6_down(s5_out))
        s7_out = self.s7_down_fusion(s7_in + self.s6_s7_down(s6_out))

        # activation
        output_features.append(self.out_module[0](s3_out))
        output_features.append(self.out_module[1](s4_out))
        output_features.append(self.out_module[2](s5_out))
        output_features.append(self.out_module[3](s6_out))
        output_features.append(self.out_module[4](s7_out))

        assert len(output_features) == 5

        return output_features

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
