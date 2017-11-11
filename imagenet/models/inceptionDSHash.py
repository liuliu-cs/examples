import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
from torch.autograd import Variable

import operator

__all__ = ['Inception3_DS_Hash', 'inception_v3_ds_hash']

dtype = torch.cuda.FloatTensor

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3_ds_hash(pretrained=False, **kwargs):
    """Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        raise NotImplementedError

    return Inception3_DS_Hash(**kwargs)


class Inception3_DS_Hash(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False):
        super(Inception3_DS_Hash, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = DynamicConv2d(3, 32, 5, kernel_size=3, stride=1, dynamic=True)
        self.Conv2d_2a_3x3 = DynamicConv2d(32, 32, 4, kernel_size=3, stride=1, dynamic=True)
        self.Conv2d_2b_3x3 = DynamicConv2d(32, 64, 4, kernel_size=3, padding=1, dynamic=True)
        self.Conv2d_3b_1x1 = DynamicConv2d(64, 80, 12, kernel_size=1, dynamic=True)
        self.Conv2d_4a_3x3 = DynamicConv2d(80, 192, 4, kernel_size=3, dynamic=True)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3  16x16x3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32 16x16x32   16x16x16
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32 15x15x32   15x15x16
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64 15x15x64
        # x = F.max_pool2d(x, kernel_size=3, stride=1)
        # 73 x 73 x 64   15x15x64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80   15x15x80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192  15x15x192
        # x = F.max_pool2d(x, kernel_size=3, stride=1)
        # 35 x 35 x 192  15x15x192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256  15x15x256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288  15x15x288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288  15x15x288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768  7x7x768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768  7x7x768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768  7x7x768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768  7x7x768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768  7x7x768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768  7x7x768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280   4x4x1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048   4x4x2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048   4x4x2048
        x = F.avg_pool2d(x, kernel_size=4)
        # 1 x 1 x 2048   1x1x2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=1)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class DynamicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool_dim, kernel_size=3, stride=1, padding=0,
                bias=None, dynamic=False, hash_size=8, num_tables=10):
        super(DynamicConv2d, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.dynamic = dynamic
        self.out_channels = out_channels
        if self.dynamic:
            self.whole_w = Variable(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size
            ), requires_grad=False)
            self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size
            ))
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        else:
            self.whole_w = Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
            self.weight = self.whole_w
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        self.pool_dim = pool_dim
        self.stride = stride
        self.padding = padding
        self.bias = bias
        # self.dynamic = dynamic

        # Hash table related
        # self.rm = Variable(torch.randn(self.L, self.K, self.P), requires_grad=False)
        # self.lsh = LSHash(8, 100, 27)
        self.hash_size = hash_size # K
        self.num_tables = num_tables # L
        self.input_dim = int(kernel_size * kernel_size * in_channels)
        self.size_limit = int(out_channels / 2)
        self.active_index = []
        self._init_random_matrix()
        self._init_hash_tables()
        self.register_buffer('one', torch.ones(1))

        self.init_weights()

    def init_weights(self):
        import scipy.stats as stats
        stddev = self.stddev if hasattr(self, 'stddev') else 0.1
        X = stats.truncnorm(-2, 2, scale=stddev)
        values = torch.Tensor(X.rvs(self.whole_w.data.numel()))
        values = values.view(self.whole_w.data.size())
        self.whole_w.data.copy_(values)
        self.weight.data.zero_()

        # init LSH table and indexing channels
        self._indexing(self.whole_w)

    def forward(self, x):
        # x = self.conv(x)
        # hashes = self.lsh.indexing(x)
        # print(hashes)
        # if not self.active_index:
        self._update_params(self.active_index)
        self.active_index = self._querying(x)
        # print(self.active_index)
        self._select_active(self.active_index)
        x = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride,
            padding=self.padding)
        # print('x: ', x.size())
        x = self.bn(x)
        return F.relu(x, inplace=True)

    def _init_random_matrix(self):
        # self.random_matrix_list = [self._generate_random_matrix()
        #                         for _ in range(0, self.num_tables)]
        self.register_buffer('random_matrix',
            torch.randn(self.num_tables, self.hash_size, self.input_dim))

    def _init_hash_tables(self):
        self.hash_tables = [ HashBucket() for _ in range(0, self.num_tables)]

    def _generate_random_matrix(self):
        return torch.randn(self.hash_size, self.input_dim).type(dtype)

    def _hashing(self, random_matrix, query):
        """ random_matrix has a dimension of 'hash_size' x 'input_dim'
            query: input vector of size 'input_dim'
            return the binary hash for query
        """
        # print('Size of random_matrix: ', random_matrix.size())
        # print(random_matrix)
        # print(query)
        # print('Size of query: ', query.data.size())
        projections = torch.matmul(random_matrix, query.data.view(-1))
        signs = torch.sign(projections)
        # print('signs: ', signs)
        return F.relu(signs)

    def _indexing(self, w):
        # print('w size: ', w.size())
        # _w = torch.chunk(w, w.size()[0], dim=0)
        # print('_w size: ', len(_w))
        # print('_w element size: ', _w[0].size())
        w = w.data.view(w.size()[0], -1)
        # print('w size: ', w.size())

        dotproduct = torch.matmul(self.random_matrix, torch.transpose(w, 0, 1))
        signs = torch.sign(dotproduct)
        hashes = F.relu(signs)
        # print('hashes size: ', hashes.size())

        hashes = torch.transpose(hashes, 1, 2)
        # print(hashes)

        # self.table = torch.zeros(self.num_tables * 256, self.out_channels).byte()
        self.table = [[] for _ in range(0, self.num_tables * 256)]

        for i in range(0, self.num_tables):
            for j in range(0, self.out_channels):
                signs = hashes[i,j,:]
                # print(signs)
                addr = self._to_signature(signs)
                # print(signature)
                self.table[i*256+addr].append(j)

    def _querying(self, x):
        # print('input size: ', x.size())
        x = F.avg_pool2d(x, self.pool_dim)
        # print('pooled size: ', x.size())
        x = x.data.view(x.size()[0], -1)
        x = torch.transpose(x, 0, 1)
        # print('random_matrix: ', self.random_matrix.size())
        # print('x size: ', x.size())
        dotproduct = torch.matmul(self.random_matrix, x)
        hashes = F.relu(torch.sign(dotproduct))
        # print('x hashes size: ', hashes.size())
        hist = {}
        active_index = []
        # for k in range(0, hashes.size()[2]):
        for i in range(0, self.num_tables):
            signs = hashes[i,:,0]
            addr = self._to_signature(signs)

            for idx in self.table[i*256+addr]:
                if idx not in hist:
                    hist[idx] = 1
                else:
                    hist[idx] += 1

        sorted_hist = sorted(hist.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_hist)
        for i in range(min(len(sorted_hist), self.size_limit)):
            active_index.append(sorted_hist.pop(0)[0])

        return active_index

    def _pooling(self, x, pool_dim):
        # x = F.avg_pool2d(t, )
        pass

    def _distance(self, x, y):
        return 1 - torch.matmul(x, y) / ((torch.matmul(x, x) * torch.matmul(y, y)) ** 0.5)

    def _select_active(self, indices):
        self.weight.data.zero_()
        for i in indices:
            self.weight.data[i,:,:,:] = self.whole_w.data[i,:,:,:]

    def _update_params(self, indices):
        for i in indices:
            self.whole_w.data[i,:,:,:] = self.weight.data[i,:,:,:]

    def _to_signature(self, x):
        signature = 0
        # print('x size ', x.size())
        # print(x)
        for i in x.split(1):
            if torch.equal(i.data, self.one): signature |= 1
            else: signature |= 0
            signature = signature << 1
        return signature>>1

class HashBucket(object):
    """docstring for HashBucket."""
    def __init__(self):
        super(HashBucket, self).__init__()
        self.keys = []
        self.values = []

    def add(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def lookup(self, key):
        value_list = []
        for i, k in enumerate(self.keys):
            if torch.equal(k.data, key.data):
                value_list.append(self.values[i])
        return value_list

    def toString(self):
        for i in range(len(self.keys)):
            print('Key: ', self.keys[i], ' Channel: ', self.values[i])
