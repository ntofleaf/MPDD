import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.autograd import Variable
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS


class FeedForward(BaseModule):
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


class FFN(BaseModule):
    """ Standard 2 layer FFN of transformer

    """
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FFN, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        # nn.init.normal_(self.linear_1.weight, std=0.001)
        # nn.init.normal_(self.linear_2.weight, std=0.001)

    def forward(self, x):
        # x = x.unsqueeze()
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(BaseModule):
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


class PositionalEncoder(BaseModule):
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
        # TODO: check if this is correct or not sqrt(2048) ~= 45.25
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


class PositionalEncoding(BaseModule):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        # seq_len = x.size(1)
        # pe = self.pe[:, :seq_len]
        x = x + self.pe
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


class TX(BaseModule):
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


class TxPositionalEncoding(BaseModule):
    def __init__(self, d_model, max_seq_len=80):
        super(TxPositionalEncoding, self).__init__()
        self.d_model = d_model

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, max_seq_len, d_model, 1, 1]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        # seq_len = x.size(1)
        # pe = self.pe[:, :seq_len]
        x = x + self.pe
        return x


class TxAttention(BaseModule):
    
    def __init__(self, channel=1024, height=7, width=7):
        super(TxAttention, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, q, k, v):
        q = q.view(-1, 1, self.channel, 1, 1)  # (16, 1, 1024, 1, 1)
        # Step 1: Expand q to match spatial dimensions of k
        q_expanded = q.expand(-1, -1, -1, self.height, self.width)  # (16, 1, 1024, 7, 7)

        # Step 2: Compute attention scores
        # attention_scores: (batch_size, num_queries, num_keys, height, width)
        # q (bs,  1, 1024, 7, 7)
        # k (bs, 32, 1024, 7, 7)
        attention_scores = torch.einsum('bnchw, bmchw -> bnmhw', q_expanded, k)  # (16, 1, 32, 7, 7)

        # Step 3: Scale the attention scores
        attention_scores = attention_scores / math.sqrt(self.channel)

        # Step 4: Apply softmax to get attention weights
        # Softmax over the num_keys dimension (dimension 2)
        attention_weights = F.softmax(attention_scores, dim=2)  # (16, 1, 32, 7, 7)

        # Step 5: Compute weighted sum over v
        # output: (batch_size, num_queries, channels, height, width)
        # a (bs,  1,   32, 7, 7)
        # v (bs, 32, 1024, 7, 7)
        output = torch.einsum('bnmhw,bmchw->bnchw', attention_weights, v)  # (16, 1, 1024, 7, 7)
        # Step 6: Aggregate spatial dimensions to get final output
        output = output.sum(dim=[3, 4], keepdim=True)  # (16, 1, 1024, 1, 1)
        output = output.squeeze(-1).squeeze(-1)  # (16, 1, 1024)
        return output


class QPrjection(BaseModule):
    def __init__(self, in_dim=2048, out_dim=1024):
        super(QPrjection, self).__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.global_avg_pool(x)
        return x.unsqueeze(1)  # (b, 1024, 1, 1)


class TxUnit(BaseModule):

    def __init__(self, t=32, d_model=1024, dropout=0.3, scale_factor=2):
        super(TxUnit, self).__init__()
        self.t = t
        self.t_proj_dim = t * scale_factor
        self.d_model = d_model
        self.k_w = nn.Conv2d(t, self.t_proj_dim, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.v_w = nn.Conv2d(t, self.t_proj_dim, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        # self.QPr = nn.Linear(d_model * 2, d_model)

        self.Tx_attention = TxAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.ff = FFN(d_model, d_ff=int(d_model / 2))

    def forward(self, q, k, v):
        """
        q: (b, 1, 2048)
        k: (b, 32, 1024, 7, 7)
        v: (b, 32, 1024, 7, 7)
        """
        # q = self.QPr(q)  # (b, 1, 1024)

        k = k.view(-1, self.t , self.d_model, 49)  # (b, 32, 1024, 49)
        v = v.view(-1, self.t, self.d_model, 49)  # (b, 32, 1024, 49)
        k = self.k_w(k)  # (b, 64, 1024, 49)
        v = self.v_w(v)  # (b, 64, 1024, 49)
        k = k.view(-1, self.t_proj_dim, self.d_model, 7, 7)  # (b, 64, 1024, 7, 7)
        v = v.view(-1, self.t_proj_dim, self.d_model, 7, 7)  # (b, 64, 1024, 7, 7)

        q_ = self.Tx_attention(q, k, v)  # (b, 1, 1024)
        q_ = self.layer_norm_1(q_ + q)
        q_new = self.layer_norm_2(q_ + self.dropout(self.ff(q_)))
        return q_new # (b, 1, 1024)


class MultiTxHead(BaseModule):
    def __init__(self, t=32, d_model=1024, dropout=0.3, heads=4):
        super(MultiTxHead, self).__init__()
        self.ffn = nn.Linear(d_model * heads, d_model)
        self.Tx_heads = nn.ModuleList(
            [TxUnit(t, d_model, dropout=dropout) for _ in range(heads)]
        )

    def forward(self, q, k, v):
        """
        q: (b, 2048, 7, 7)
        """
        outputs = []
        for Tx_head in self.Tx_heads:
            outputs.append(Tx_head(q, k, v))  # (b, 1, 1024)
        x = torch.cat(outputs, -1)  # (b, 4, 1024)
        x = self.ffn(x)
        return x


class MultiTxLayer(BaseModule):
    def __init__(self, t, d_model=1024, dropout=0.3):
        super(MultiTxLayer, self).__init__()
        self.Tx_layer_1 = MultiTxHead(t, d_model, dropout=dropout)
        self.Tx_layer_2 = MultiTxHead(t, d_model, dropout=dropout)
        self.Tx_layer_3 = MultiTxHead(t, d_model, dropout=dropout)
        # self.Tx_layer_4 = MultiTxHead(d_model, dropout=dropout)
        # self.Tx_layer_5 = MultiTxHead(d_model, dropout=dropout)
        # self.Tx_layer_6 = MultiTxHead(d_model, dropout=dropout)

    def forward(self, q, k, v):
        # multi-layer attention
        # q: (b, 1, 2048)
        q = self.Tx_layer_1(q, k, v)  # (b, 1, 2048)
        q = self.Tx_layer_2(q, k, v)  # (b, 1, 2048)
        q = self.Tx_layer_3(q, k, v)  # (b, 1, 2048)
        # q = self.Tx_layer_4(q, k, v)  # (b, 1, 2048)
        # q = self.Tx_layer_5(q, k, v)  # (b, 1, 2048)
        # q = self.Tx_layer_6(q, k, v)  # (b, 1, 2048)
        return q


class ExpAttention(BaseModule):

    def __init__(self, d_model, d_k, dropout=0.3):
        super(ExpAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.q_w = nn.Linear(d_model, d_k)
        self.k_w = nn.Linear(d_model, d_k)
        self.v_w = nn.Linear(d_model, d_k)
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = Norm(d_k)
        self.norm_2 = Norm(d_k)
        self.ff = FeedForward(d_k, d_ff=int(d_k * 4))

    def forward(self, q, k, v, mask=None):
        q = self.q_w(q)
        k = self.k_w(k)
        v = self.v_w(v)
        a = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        a = torch.softmax(a, dim=-1)
        a = torch.matmul(a, v)
        q_ = self.norm_1(a + q)
        new_query = self.norm_2(q_ + self.dropout(self.ff(q_)))
        return new_query


class MultiHeadExpAttention(BaseModule):

    def __init__(self, d_model=1024, head=8, dropout=0.3):
        super(MultiHeadExpAttention, self).__init__()
        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.ff = FeedForward(self.d_model, d_ff=int(d_model * 2))
        self.head_layers = []
        for i in range(self.head):
            self.head_layers.append(ExpAttention(d_model, self.d_k))
        self.heads_list = nn.ModuleList(self.head_layers)

    def forward(self, q, k, v):
        # multihead attention
        outputs = []
        for i in range(self.head):
            outputs.append(self.heads_list[i](q, k, v))
        q_cat = torch.cat(outputs, -1)
        q_new = self.linear(q_cat)
        q = self.norm_1(q_new + q)

        # Feed Forward
        f = q + self.norm_2(self.dropout(self.ff(q)))
        return f


class BlockHead(BaseModule):
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


@MODELS.register_module()
class VatNeck(BaseModule):
    def __init__(self, seq_len, head=16, spatial_h=4, spatial_w=4):
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


@MODELS.register_module()
class VatNeckA(BaseModule):

    def __init__(self, seq_len, head=2, spatial_h=4, spatial_w=4, num_features=2048):
        super(VatNeckA, self).__init__()
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.num_features = num_features
        self.num_frames = seq_len
        self.d_model = int(self.num_features / 2)

        # reduce the dimensionality of the input
        self.dimension_proj = nn.Sequential(
            nn.Conv2d(
                self.num_features, self.d_model, kernel_size=(1, 1), stride=1, padding=0, bias=False,
            ),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
        )
        # feature alignment which simulate the RoIPooling layer as in the paper
        self.feature_alignment = nn.Sequential(
            nn.Conv2d(
                self.d_model, self.d_model, kernel_size=(7, 7), stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
        )
        # num_frames is the sequence length
        self.pos_embedding = TxPositionalEncoding(self.d_model, self.num_frames)
        # TxUnit is the core of the model 
        self.multi_Tx_layer = MultiTxLayer(t=seq_len, d_model=self.d_model)

    def forward(self, x, b, t):
        x = self.dimension_proj(x)
        # simulate the trunk output as in the paper
        trunk = x.view(-1, t, self.d_model, self.spatial_h, self.spatial_w)
        trunk = self.pos_embedding(trunk)
        # q vector from the middle frame of the sequence
        # q [b, 1024, 7, 7] --> [b, 2048, 1, 1]
        # feature alignment which simulate the RoIPooling layer as in the paper
        q = self.feature_alignment(trunk[:, int(t / 2), :, :]).view(b, 1, -1)
        v = trunk  # value
        k = trunk  # key

        f = self.multi_Tx_layer(q, k, v)

        return f.squeeze(1)  # (b, 2048)


@MODELS.register_module()
class EatNeck(BaseModule):
    def __init__(self, seq_len, head=16, spatial_h=4, spatial_w=4, num_features=2048):
        super(EatNeck, self).__init__()
        # self.spatial_h = 7
        self.spatial_h = spatial_h
        # self.spatial_w = 4
        self.spatial_w = spatial_w
        self.head = head
        self.num_features = num_features
        self.num_frames = seq_len
        self.d_model = int(self.num_features / 2)
        self.d_k = self.d_model // self.head
        self.bn1 = nn.BatchNorm2d(self.d_model)
        self.bn2 = Norm(self.d_model, trainable=False)

        self.pos_embedding = PositionalEncoding(self.d_model, self.num_frames)
        self.conv_pr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        # self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(7, 4), stride=1, padding=0, bias=False)
        self.Qpr = nn.Conv2d(
            self.d_model,
            self.d_model,
            kernel_size=(self.spatial_h, self.spatial_w),
            stride=1,
            padding=0,
            bias=False)
        # self.Qpr = nn.AdaptiveAvgPool2d(1)
        self.multihead = MultiHeadExpAttention(self.d_model, self.head)
        self.multihead2 = MultiHeadExpAttention(self.d_model, self.head)
        self.multihead3 = MultiHeadExpAttention(self.d_model, self.head)
        # self.head_layers = []
        # for i in range(self.head):
        #     self.head_layers.append(BlockHead())

        # self.list_layers = nn.ModuleList(self.head_layers)
        # # self.classifier = nn.Linear(self.d_model, num_classes)
        # # resnet style initialization
        # # nn.init.kaiming_normal_(self.Qpr.weight, mode='fan_out')
        # # nn.init.normal_(self.classifier.weight, std=0.001)
        # # nn.init.constant(self.classifier.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, b, t):
        x = self.Qpr(self.conv_pr(x))
        x = self.bn1(x)
        x = x.view(-1, t, self.d_model)
        # x = x.view(-1, t, self.d_model)
        # stabilizes the learning
        # x = x.view(b, t, self.num_features, self.spatial_h, self.spatial_w)
        x = self.pos_embedding(x)
        # x = x.view(-1, self.num_features, self.spatial_h, self.spatial_w)
        # x = F.relu(self.Qpr(x))
        # x: (b,t,1024,1,1) since its a convolution: spatial positional encoding is not added
        # paper has a different base (resnet in this case): which 2048 x 7 x 4 vs 16 x 7 x 7
        x = self.bn2(x)
        # stabilization
        q = x[:, int(t / 2): int( t / 2) + 1, :]  # middle frame is the query
        v = x  # value
        k = x  # key

        # q = q.view(b, self.head, self.d_k)
        # k = k.view(b, t, self.head, self.d_k)
        # v = v.view(b, t, self.head, self.d_k)

        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)
        # #  q: b, 16, 64
        # #  k,v: b, 16, 10 ,64
        # outputs = []
        # for i in range(self.head):
        #     outputs.append(self.list_layers[i](q[:, i], k[:, i], v[:, i]))

        # f = torch.cat(outputs, 1)
        # f = F.normalize(f, p=2, dim=1)
        f = self.multihead(q, k, v)
        f = self.multihead2(f, k, v)
        f = self.multihead3(f, k, v)

        return f.squeeze(1)

