import torch
import torch.nn.functional as F
from torch import nn
import math
import torchvision
from torch.autograd import Variable
from typing import Optional, Tuple, List


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


class VatBackbone(nn.Module):
    """ Base is resnet tail is the main transformer network


    """
    def __init__(self, init_weights=False, vggface_pretrain=None):  # seq_len --> num_frames
        super(VatBackbone, self).__init__()
        
        resnet50 = torchvision.models.resnet50(pretrained=True)
            
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        return x, b, t


class FeedForward(nn.Module):
    """ Standard 2 layer FFN of transformer

    """
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.normal_(self.linear_1.weight, std=0.001)
        nn.init.normal_(self.linear_2.weight, std=0.001)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    """ Standard NORM layer of Transformer

    """
    def __init__(self, d_model, eps=1e-6, trainable=False):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):
    """ Standard positional encoding (addition/ concat both are valid)

    """
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = z.expand(batch_size, seq_len, num_feature, spatial_h, spatial_w)
        x = x + z
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    # standard attention layer
    scores = torch.sum(q * k, -1) / math.sqrt(d_k)
    # scores : b, t
    scores = F.softmax(scores, dim=-1)
    scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
    # scores : b, t, dim
    output = scores * v
    output = torch.sum(output, 1)
    if dropout:
        output = dropout(output)
    return output


class TX(nn.Module):
    def __init__(self, d_model=64, dropout=0.3):
        super(TX, self).__init__()
        self.d_model = d_model
        # no of head has been modified to encompass : 1024 dimension
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff=int(d_model / 2))

    def forward(self, q, k, v, mask=None):
        # q: (b , dim )
        b = q.size(0)
        t = k.size(1)
        dim = q.size(1)
        q_temp = q.unsqueeze(1)
        q_temp = q_temp.expand(b, t, dim)
        # q,k,v : (b, t , d_model=1024 // 16 )
        A = attention(q_temp, k, v, self.d_model, mask, self.dropout)
        # A : (b , d_model=1024 // 16 )
        q_ = self.norm_1(A + q)
        new_query = self.norm_2(q_ + self.dropout_2(self.ff(q_)))
        return new_query


class BlockHead(nn.Module):
    def __init__(self, d_model=64, dropout=0.3):
        super(BlockHead, self).__init__()
        self.T1 = TX()
        self.T2 = TX()
        self.T3 = TX()

    def forward(self, q, k, v, mask=None):
        q = self.T1(q, k, v)
        q = self.T2(q, k, v)
        q = self.T3(q, k, v)
        return q


class VatNeck(nn.Module):
    def __init__(self, seq_len=32, head=16, spatial_h=7, spatial_w=7):
        super(VatNeck, self).__init__()
        # self.spatial_h = 7
        self.spatial_h = spatial_h
        # self.spatial_w = 4
        self.spatial_w = spatial_w
        self.head = head
        self.num_features = 2048
        self.num_frames = seq_len
        self.d_model = int(self.num_features / 2)
        self.d_k = self.d_model // self.head
        self.bn1 = nn.BatchNorm2d(self.num_features)
        self.bn2 = Norm(self.d_model, trainable=False)

        self.pos_embedding = PositionalEncoder(self.num_features, self.num_frames)
        # self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(7, 4), stride=1, padding=0, bias=False)
        self.Qpr = nn.Conv2d(
            self.num_features,
            self.d_model,
            kernel_size=(self.spatial_h, self.spatial_w),
            stride=1,
            padding=0,
            bias=False)

        self.head_layers = []
        for i in range(self.head):
            self.head_layers.append(BlockHead())

        self.list_layers = nn.ModuleList(self.head_layers)
        # self.classifier = nn.Linear(self.d_model, num_classes)
        # resnet style initialization
        nn.init.kaiming_normal_(self.Qpr.weight, mode='fan_out')
        # nn.init.normal_(self.classifier.weight, std=0.001)
        # nn.init.constant(self.classifier.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, b, t):
        x = self.bn1(x)
        # stabilizes the learning
        x = x.view(b, t, self.num_features, self.spatial_h, self.spatial_w)
        x = self.pos_embedding(x)
        x = x.view(-1, self.num_features, self.spatial_h, self.spatial_w)
        x = F.relu(self.Qpr(x))
        # x: (b,t,1024,1,1) since its a convolution: spatial positional encoding is not added
        # paper has a different base (resnet in this case): which 2048 x 7 x 4 vs 16 x 7 x 7
        x = x.view(-1, t, self.d_model)
        x = self.bn2(x)
        # stabilization
        q = x[:, int(t / 2), :]  # middle frame is the query
        v = x  # value
        k = x  # key

        q = q.view(b, self.head, self.d_k)
        k = k.view(b, t, self.head, self.d_k)
        v = v.view(b, t, self.head, self.d_k)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        #  q: b, 16, 64
        #  k,v: b, 16, 10 ,64
        outputs = []
        for i in range(self.head):
            outputs.append(self.list_layers[i](q[:, i], k[:, i], v[:, i]))

        f = torch.cat(outputs, 1)
        f = F.normalize(f, p=2, dim=1)

        return f


class LinearClsRegHead(nn.Module):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 in_channels: int = 1024,
                 num_classes: int = 4,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsRegHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.reg_fc = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.cls_fc = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        # self.com_fc = nn.Linear(self.in_channels, 512)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        reg_score = self.reg_fc(pre_logits)
        # com_embed = self.cls_fc[:2](pre_logits)
        cls_score = self.cls_fc(pre_logits)
        cls_score = torch.sigmoid(cls_score)
        return reg_score, cls_score


class ConvTransModel(nn.Module):

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(ConvTransModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def extract_feat(self, inputs):

        x = self.backbone(inputs)
        x = self.neck(*x)
        return self.head.pre_logits(x)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List] = None,
                **kwargs) -> List:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head(feats)


if __name__ == "__main__":
    
    model = ConvTransModel(backbone=VatBackbone(), neck=VatNeck(), head=LinearClsRegHead())
    # load_checkpoint(
    #     model, 
    #     'saved_model/20240416_173204_rmse_7.76/0416_epoch_25_rmse7.76_cls0.62.pth',
    # )
    state_dict = torch.load(
        "saved_model/20240416_173204_rmse_7.76/0416_epoch_25_rmse7.76_cls0.62.pth"
    )["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    # print(model)
    model.eval()
    model.cuda()
    inputs = torch.ones(4, 32, 3, 224, 224).float()
    inputs = inputs.cuda()
    outputs = model(inputs)
    print(outputs)

    