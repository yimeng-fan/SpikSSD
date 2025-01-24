import torch.nn as nn

from spikingjelly.activation_based import neuron

from .EMS_ResNet import ems_resnet18, ems_resnet34
from .MS_ResNet import ms_resnet18, ms_resnet34
from .MDS_ResNet import mds_resnet10, mds_resnet18, mds_resnet34
from .spiking_densenet import *
from .SEW_ResNet import *


def get_model(args):
    norm_layer = nn.BatchNorm2d if args.bn else None
    ms_neuron = neuron.ParametricLIFNode

    family, version = args.model.split('-')
    if family == "densenet":
        depth, growth_rate = version.split('_')
        blocks = {"121": [6, 12, 24, 16], "169": [6, 12, 32, 32]}
        return multi_step_spiking_densenet_custom(
            2 * args.tbin, norm_layer=norm_layer,
            multi_step_neuron=ms_neuron,
            growth_rate=int(growth_rate), block_config=blocks[depth],
            num_classes=2, backend="cupy", step_mode='m'
        )

    elif family == "msresnet":
        if int(version) == 18:
            return ms_resnet18(num_classes=2, backend='cupy')
        elif int(version) == 34:
            return ms_resnet34(num_classes=2, backend='cupy')

    elif family == "emsresnet":
        if int(version) == 18:
            return ems_resnet18(num_classes=2, backend='cupy')
        elif int(version) == 34:
            return ems_resnet34(num_classes=2, backend='cupy')

    elif family == "mdsresnet":
        if int(version) == 10:
            return mds_resnet10(num_classes=2, backend='cupy', fusion=args.fusion, attention=args.attention)
        elif int(version) == 18:
            return mds_resnet18(num_classes=2, backend='cupy', fusion=args.fusion, attention=args.attention)
        elif int(version) == 34:
            return mds_resnet34(num_classes=2, backend='cupy', fusion=args.fusion, attention=args.attention)

    elif family == "sewresnet":
        if int(version) == 18:
            return sew_resnet18(num_classes=2, connect_f=args.connect_f, zero_init_residual=True, backend='cupy')
        elif int(version) == 34:
            return sew_resnet34(num_classes=2, connect_f=args.connect_f, zero_init_residual=True, backend='cupy')
        elif int(version) == 50:
            return sew_resnet50(num_classes=2, connect_f=args.connect_f, zero_init_residual=True, backend='cupy')
        elif int(version) == 101:
            return sew_resnet101(num_classes=2, connect_f=args.connect_f, zero_init_residual=True, backend='cupy')
