"""
code modified from https://github.com/ppriyank/Video-Action-Transformer-Network-Pytorch-.git
"""
import torch
from torch import nn
import torchvision
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.models.necks.vat import VatNeck


def initialize_weights(model):
    for m in model.modules():
        # normal conv layers
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # normal BN layers
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # normal FC layers
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


@MODELS.register_module()
class SemiTransformer(BaseModule):
    """ Base is resnet tail is the main transformer network


    """
    def __init__(self, seq_len, init_weights=True, spatial_h=4, spatial_w=4):  # seq_len --> num_frames
        super(SemiTransformer, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.neck = VatNeck(seq_len, spatial_h=spatial_h, spatial_w=spatial_w)
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        # x: (b,t,2048,7,4)
        return self.neck(x, b, t)


@MODELS.register_module()
class VatBackbone(BaseModule):
    """ Base is resnet tail is the main transformer network


    """
    def __init__(self, init_weights=False, vggface_pretrain=None):  # seq_len --> num_frames
        super(VatBackbone, self).__init__()
        if vggface_pretrain:
            from .resnet_vggface import resnet50, load_state_dict
            resnet50 = resnet50(num_classes=8631, include_top=True)
            resnet50 = load_state_dict(resnet50, vggface_pretrain)
        else:
            # resnet50 = torchvision.models.resnet50(pretrained=True)
            # mobilefacenet = torchvision.models.mobilenet_v2(pretrained=True)
            shufflenet = torchvision.models.shufflenet_v2_x2_0(
                weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
            )
            # shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained=True)

        # self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.base = nn.Sequential(*list(shufflenet.children())[:-1])
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        return x, b, t


if __name__ == "__main__":
    vat_model = SemiTransformer(num_classes=1, seq_len=32).cuda()
    x_in = torch.randn(4, 32, 3, 112, 112).cuda()
    y_out = vat_model(x_in)
    print(y_out.shape)
