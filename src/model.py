import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, roc_curve
from torch.nn import ReLU, Sigmoid, Softmax
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np


class KGHeteroGNN_Initial(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('kg_entities', 'to', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
            if i!= num_layers-1:
                # hidden_channels = 256
                hidden_channels = int(hidden_channels/2)

        self.lin = Linear(hidden_channels, out_channels)
        # self.lin1 = Linear(hidden_channels, hidden_channels//2)
        # self.lin = Linear(hidden_channels//2, out_channels)
        # self.sigmoid = Sigmoid()

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        # out = self.sigmoid(self.lin1(x_dict["news"]))
        return self.lin(x_dict['news'])
        # return self.lin(out)

def train(model, data, args):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data['news'].train_mask
        #print(out[mask].shape, data['news'].y[mask])
        loss = criterion(out[mask], data['news'].y[mask])
        loss.backward()
        optimizer.step()

def test(model, data, args, i):
    out = model(data.x_dict, data.edge_index_dict)
    pred = out[data['news'].test_mask].argmax(dim=1).cpu()

    y = data['news'].y[[data['news'].test_mask]].cpu()
    # pred_list = out[data['news'].test_mask].tolist()
    # predict = []
    def softmax(p):
        e_x = torch.exp(p)
        partition_x = e_x.sum(1, keepdim=True)
        return e_x / partition_x
    predict = softmax(out[data['news'].test_mask])
    col, row = predict.shape
    print(col)
    pred_list = []
    for i in range(col):
        pred_list.append(predict[i][1].cpu().tolist())
    pred_list = torch.Tensor(pred_list)


    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, )
    # f1 score
    f1 = f1_score(y, pred)
    recall = recall_score(y, pred,)
    # f1_1 = f1_score(y, pred, average='weighted')
    # f1_2  = f1_score(y, pred, pos_label=0)

    auc = roc_auc_score(y, pred,)
    print(f"Testing Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},F1: {f1:.4f}")
    
