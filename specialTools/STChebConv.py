import torch
from torch import nn
from torch.nn import Parameter
from torch_sparse import spmm
from torch_geometric.utils import degree, remove_self_loops

from torch_geometric.nn.inits import uniform
from lxyTools.pytorchTools import multilayerSelfAttention


class STChebConv(torch.nn.Module):

    def __init__(self, qk_dim, out_channels, in_channels, num_layers, K, bias=True):
        super(STChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.K = K

        self.grulist = nn.ModuleList([multilayerSelfAttention(qk_dim, out_channels, in_channels, num_layers, 64) for i in range(K)])

        self.weight = Parameter(torch.Tensor(K, 144*in_channels, 144*out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels*144))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * 144 * self.K

        uniform(size, self.weight)
        uniform(size, self.bias)


    def conv_out(self, x, k_i):

        # x = x.contiguous().view(-1, 144, self.in_channels)
        # out = self.grulist[k_i](x)
        # out = out.contiguous().view(-1, 144*self.out_channels)
        return x

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(0), row.size(0), self.K

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges, ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        outlist = []

        # Perform filter operation recurrently.
        Tx_0 = x
        out = torch.mm(self.conv_out(Tx_0, 0), self.weight[0])
        outlist.append(out)
        # out = torch.mm(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm(edge_index, lap, num_nodes, x)
            # out = out + torch.mm(Tx_1, self.weight[1])
            # out = out + torch.mm(self.conv_out(Tx_1, 1), self.weight[1])
            out = torch.mm(self.conv_out(Tx_1, 1), self.weight[1])
            outlist.append(out)

        for k in range(2, K):
            Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
            # out = out + torch.mm(Tx_2, self.weight[k])
            # out = out + torch.mm(self.conv_out(Tx_2, k), self.weight[k])
            out = torch.mm(self.conv_out(Tx_2, k), self.weight[k])
            outlist.append(out)
            Tx_0, Tx_1 = Tx_1, Tx_2

        out = torch.stack(outlist, dim=0)
        out = torch.sum(out, dim=0)

        if self.bias is not None:
            out = out + self.bias

        return out
