import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

import time


from lxyTools.pytorchTools import selfAttention
from lxyTools.pytorchTools import GatedConv1d

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import APPNP
from specialTools.GatedGraphConv import GatedGraphConv

from specialTools.STChebConv import STChebConv

from torch_geometric.data import Batch
from lxyTools.pytorchTools import set_random_seed


set_random_seed(2018)

raw_train = pd.read_csv('./processedData/train.csv')
test = pd.read_csv('./processedData/test.csv')


roadmap = pd.read_csv('./rawdata/Metro_roadMap.csv')
#
# roadmap.loc[roadmap.index==55, '53'] = 1
# roadmap.loc[np.array(roadmap['53']==1)&np.array(roadmap.index==54), '53'] = 0
#
# roadmap.loc[roadmap.index==53, '55'] = 1
# roadmap.loc[np.array(roadmap['55']==1)&np.array(roadmap.index==54), '55'] = 0
#
# roadmap.loc[roadmap['54']==1, '54'] = 0

max_range = np.percentile(np.sqrt(raw_train.inNums), 90)


# train_set_days = [4, 5, 6, 11, 12, 13, 18, 19]
train_set_days = [5, 6, 12, 13, 19, 26]
# train_set_days = [4, 6, 11, 13, 18, 25]
# train_set_days = [4, 5, 11, 12, 13, 18, 19, 25, 26]
valid_set_days = [20]


train = raw_train[raw_train.day.apply(lambda x: x in train_set_days)]
valid = raw_train[raw_train.day.apply(lambda x: x in valid_set_days)]


index_list1=[]
index_list2=[]
for i in range(roadmap.shape[0]):
    poslist = roadmap[roadmap[str(i)]==1].index.tolist()
    for pos in poslist:
        index_list1.append(i)
        index_list2.append(pos)
edge_index = torch.tensor([index_list1, index_list2], dtype=torch.long).cuda()

node_feature_names = train.drop(['stationID', 'time', 'day', 'inNums', 'outNums'], axis=1).columns.tolist()

# out_features_index = []
# in_features_index = []
# for i in range(len(node_feature_names)):
#     if 'out' in  node_feature_names[i]:
#         out_features_index.append(i)
#     if 'in' in node_feature_names[i]:
#         in_features_index.append(i)




class mydataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list
        self.transform = None
    def __len__(self):
        return len(self.graph_list)
    def get(self, idx):
        return self.graph_list[idx]



# def Df_to_Graph(raw_data, train_set_days, valid_set_days, test_set_days):
#     train_graph_list = []
#     valid_graph_list = []
#     test_graph_list = []
#     for day in (train_set_days+valid_set_days+test_set_days):
#         train_day = raw_data[np.array(raw_data.day==day)|np.array(raw_data.day==day-7)]
#         print(day)
#         print(train_day.shape)
#         print(train_day.head())
#         train_day = train_day.sort_values(by=['stationID', 'time', 'day'])
#
#         data_content = train_day[node_feature_names].values.reshape(-1, 144*2*len(node_feature_names))
#         data_content = np.sqrt(data_content)/max_range
#         data_content = torch.tensor(data_content, dtype=torch.float).cuda()
#
#
#         label = train_day[train_day.day==day][['outNums', 'inNums']].values.reshape(-1, 144*2)
#         label = np.sqrt(label)/max_range
#         label = torch.tensor(label, dtype=torch.float).cuda()
#
#
#         if data_content.shape[0] != 81:
#             print('err')
#             while True: pass
#
#         single_graph = Data(x=data_content, y=label, edge_index=edge_index)
#
#         if day in train_set_days:
#             train_graph_list.append(single_graph)
#         if day in valid_set_days:
#             valid_graph_list.append(single_graph)
#         if day in test_set_days:
#             test_graph_list.append(single_graph)
#
#     return mydataset(train_graph_list), mydataset(valid_graph_list), mydataset(test_graph_list)
#
# dataset_train, dataset_valid, dataset_test = Df_to_Graph(pd.concat([raw_train, test[raw_train.columns]], axis=0), train_set_days, valid_set_days, [29])


# train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
# valid_loader = DataLoader(dataset_valid, batch_size=1, shuffle=False)
# test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)



def Df_to_Graph(data_df):
    graph_list = []
    for day in data_df.day.unique():
        train_day = data_df[data_df.day==day]
        train_day = train_day.sort_values(by=['stationID', 'time'])

        data_content = train_day[node_feature_names].values.reshape(-1, 144*len(node_feature_names))
        data_content = np.sqrt(data_content)/max_range
        data_content = torch.tensor(data_content, dtype=torch.float).cuda()


        label = train_day[['outNums', 'inNums']].values.reshape(-1, 144*2)
        label = np.sqrt(label)/max_range
        label = torch.tensor(label, dtype=torch.float).cuda()


        if data_content.shape[0] != 81:
            print('err')
            while True: pass

        single_graph = Data(x=data_content, y=label, edge_index=edge_index)

        graph_list.append(single_graph)
    return mydataset(graph_list)


train_loader = DataLoader(Df_to_Graph(train), batch_size=1, shuffle=True)
valid_loader = DataLoader(Df_to_Graph(valid), batch_size=1, shuffle=False)
test_loader = DataLoader(Df_to_Graph(test), batch_size=1, shuffle=False)





class MultiScaleConv(torch.nn.Module):
    def __init__(self, in_channels=8, out_channels=2):
        super(MultiScaleConv, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv1d_k25 = nn.Conv1d(in_channels, out_channels, kernel_size=25, padding=12).cuda()
        self.conv1d_k13 = nn.Conv1d(in_channels, out_channels, kernel_size=13, padding=6).cuda()
        self.conv1d_k7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3).cuda()
        self.conv1d_k3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1).cuda()

    def forward(self, x):

        x = x.contiguous().view(-1, 144, self.in_channels).transpose(1, 2).contiguous()
        conv1d_k25 = self.conv1d_k25(x)
        conv1d_k13 = self.conv1d_k13(x)
        conv1d_k7 = self.conv1d_k7(x)
        conv1d_k3 = self.conv1d_k3(x)
        conv1d = torch.cat([conv1d_k25, conv1d_k13, conv1d_k7, conv1d_k3], 1)

        conv1d_cat = torch.cat([conv1d, x], 1)

        conv1d = conv1d.contiguous().transpose(1, 2).contiguous().view(-1, 144*(self.out_channels*4))
        conv1d_cat = conv1d_cat.contiguous().transpose(1, 2).contiguous().view(-1, 144*(self.out_channels*4 + self.in_channels))

        return conv1d_cat, conv1d

class MultiScaleChebConv(torch.nn.Module):
    def __init__(self, in_channels=8, out_channels=2):
        super(MultiScaleChebConv, self).__init__()

        out_channels *= 144
        in_channels *= 144

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.chebconv_k3 = ChebConv(in_channels, out_channels, K=4).cuda()
        self.chebconv_k5 = ChebConv(in_channels, out_channels, K=5).cuda()
        self.chebconv_k7 = ChebConv(in_channels, out_channels, K=6).cuda()
        self.chebconv_k9 = ChebConv(in_channels, out_channels, K=7).cuda()

    def forward(self, x, edge_index):

        chebconv_k3 = self.chebconv_k3(x, edge_index)
        chebconv_k5 = self.chebconv_k5(x, edge_index)
        chebconv_k7 = self.chebconv_k7(x, edge_index)
        chebconv_k9 = self.chebconv_k9(x, edge_index)
        chebconv = torch.cat([chebconv_k3, chebconv_k5, chebconv_k7, chebconv_k9], 1)
        # chebconv_cat = torch.cat([conv1d, x], 1)

        return chebconv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.dropout = nn.Dropout(0.10)
        # self.st_conv = STChebConv(8, 8, 8, 3, K=5).cuda()

        self.multi_conv1d1 = MultiScaleConv(8, 2).cuda()

        # self.conv1 = MultiScaleChebConv(16, 2).cuda()
        self.conv1 = ChebConv(144*8, 144*8, K=6).cuda()
        self.conv2 = ChebConv(144*8, 144*4, K=5).cuda()
        self.conv3 = ChebConv(144*4, 144*2, K=5).cuda()

        # self.multi_conv1d2 = MultiScaleConv(8, 2).cuda()



        self.selu = nn.SELU()
        self.relu = nn.ReLU()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        x = self.dropout(x)

        # multi_conv1d_1_cat, multi_conv1d_1  = self.multi_conv1d1(x)
        # multi_conv1d_1_cat = self.relu(multi_conv1d_1_cat)
        # multi_conv1d_1 = self.relu(multi_conv1d_1)

        x_conv1 = self.conv1(x, edge_index)
        x_conv1 = self.relu(x_conv1)

        # x_conv1 = self.linear(x_conv1)
        # x_conv1 = self.relu(x_conv1)

        # x_conv2 = x_conv1
        # for conv in self.conv2:
        #     x_conv2 = conv(x_conv2, edge_index)

        # multi_conv1d_2_cat, multi_conv1d_2 = self.multi_conv1d2(x_conv1)


        x_conv2 = self.conv2(x_conv1, edge_index)
        x_conv2 = self.relu(x_conv2)

        x_conv3 = self.conv3(x_conv2, edge_index)
        x_conv3 = self.relu(x_conv3)

        return x_conv3

set_random_seed(2018)
model = Net()


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,1500,2000], gamma=0.1)

loss_fn = nn.L1Loss()

epoch_nums = 2200

for epoch in range(epoch_nums):
    start_time = time.time()

    scheduler.step()

    model.train()
    avg_loss = 0
    for batch in train_loader:
        pred = model(batch)

        loss = loss_fn(pred**2, batch.y**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()*(max_range**2)*batch.num_graphs / len(train_loader)

    for batch in valid_loader:
        pred = model(batch)

        loss = loss_fn(pred**2, batch.y**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()*(max_range**2)*batch.num_graphs / len(valid_loader)

    model.eval()
    avg_val_loss = 0
    valid_loss_list = []
    for batch in valid_loader:
        pred = model(batch)
        pred = pred.detach()

        loss = loss_fn(pred**2, batch.y**2)

        valid_loss_list.append(loss.detach().cpu().numpy()*(max_range**2))

        avg_val_loss += loss.item()*(max_range**2)*batch.num_graphs / len(valid_loader)

    end_time = time.time()
    elapsed_time = end_time - start_time

    str_out = '\t'
    for val in valid_loss_list:
        str_out += 'valid_loss:\t'+str(val)[0:5]+'\t'
    # print(str_out)


    print('Epoch {}/{} \t loss={:.4f}  \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, epoch_nums, avg_loss, avg_val_loss, elapsed_time)+str_out)




test_pred_list = []
for batch in test_loader:
    pred = model(batch)
    pred = pred.detach().cpu().numpy()

    pred = (pred**2)*(max_range**2)

    test_pred_list.append(pred)

test_pred = np.array(test_pred_list)
test_pred = test_pred.reshape(81*144, 2)


test_pred_df = pd.DataFrame({'outNums': test_pred[:, 0], 'inNums': test_pred[:, 1]})

test_pred_df[['stationID', 'time']] = test.sort_values(by=['stationID', 'time'])[['stationID', 'time']]


out_df = pd.merge(test.drop(['inNums', 'outNums'], axis=1), test_pred_df, on=['stationID', 'time'], how='left')


out_df[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('./submit.csv', index=False)

plt.figure(figsize=(12, 8))
plt.subplot(221)
sns.distplot(out_df.outNums)
plt.subplot(222)
sns.distplot(raw_train.outNums)


plt.subplot(223)
sns.distplot(out_df.inNums)
plt.subplot(224)
sns.distplot(raw_train.inNums)

plt.show()
