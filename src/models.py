
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from layers import GraphAttentionLayer,GraphConvolution,SageConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)




class GCN_En(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_En2(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_Classifier(nn.Module):
    
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


class Sage_En(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_En2(nn.Module):
    
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En2, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_Classifier(nn.Module):

    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


"""
class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout,alpha, nheads=4):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, int(nhid / nheads), dropout=dropout,alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)

    def forward(self, x, adj):

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.elu(self.linear(x))
        x = F.elu(x)
        return x
"""

class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout,alpha, nheads=4):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, int(nhid / nheads), dropout=dropout,alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)

    def forward(self, x, adj):

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.elu(self.linear(x))
        x = F.elu(x)
        return x

class GAT_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, alpha, nheads=1):
        super(GAT_En2, self).__init__()
        ## https: // zhuanlan.zhihu.com / p / 128072201
        self.attentions_1 = [GraphAttentionLayer(in_features = nfeat,out_features =  int(nhid/nheads) ,alpha=alpha, dropout=dropout,concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention1_{}'.format(i), attention)

        self.linear_1 = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.attentions_2 = [GraphAttentionLayer(in_features = nhid, out_features = int(nembed / nheads),alpha=alpha, dropout=dropout,concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention2_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(in_features = nhid*nheads, out_features = nembed , dropout=dropout, alpha=alpha, concat=False)

        self.linear_2 = nn.Linear(nembed, nembed)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_1.weight, std=0.05)
        nn.init.normal_(self.linear_2.weight, std=0.05)


    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions_1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)   # concat = True 已经包含激活函数了

        return x


class GAT_official(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_official, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x, F.log_softmax(x, dim=1)


class GAT_Classifier(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads=8):
        super(GAT_Classifier, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, int(nhid / nheads),alpha=alpha, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_classifier = GraphAttentionLayer(nhid, nclass,alpha=alpha, dropout=dropout)

        self.dropout = dropout

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)  # pyGAT是这么写的有两个dropout
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_classifier(x, adj))
        return x

class Decoder(nn.Module):

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out



class Classifier(nn.Module):
    def __init__(self, nembed,  nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)

        return x


class Decoder_feature(nn.Module):
    def __init__(self, nembed, nhid, nfeat, dropout):
        super(Decoder_feature, self).__init__()
        self.mlp1 = nn.Linear(nembed,nhid)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(nhid,nfeat)
        self.dropout = nn.Dropout(p=dropout)  # dropout训练
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight, std=0.05)
        nn.init.normal_(self.mlp2.weight, std=0.05)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.mlp2(out)
        return out


class GAT_Discriminator(nn.Module):
    def __init__(self, nembed, nhid,  dropout, alpha, nheads=1, nclass=7):
        super(GAT_Discriminator, self).__init__()
        ## https: // zhuanlan.zhihu.com / p / 128072201
        self.attentions_1 = [
            GraphAttentionLayer(in_features=nembed, out_features=int(nhid / nheads), alpha=alpha, dropout=dropout) for _
            in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention1_{}'.format(i), attention)
        self.dropout = dropout

        self.mlp1 = nn.Linear(nhid,2)


        self.classifier = GraphAttentionLayer(in_features=nhid, out_features=nclass,alpha=alpha, dropout=dropout)
        #self.fakereal = GraphAttentionLayer(in_features=nhid,out_features=2,alpha=alpha, dropout=dropout)
        
        #self.classifier = nn.Linear(in_features=nhid, out_features=nclass)
        #self.fakereal = nn.Linear(in_features=nhid, out_features=2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight, std=0.05)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        logits = F.elu(self.classifier(x,adj))
        fakeorreal = F.elu(self.mlp1(x))

        x_class = F.log_softmax(logits,dim=1)    
        x_fakereal = F.log_softmax(fakeorreal,dim=1)   

        return logits, fakeorreal, x_class, x_fakereal


class GCN_Discriminator(nn.Module):

    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Discriminator, self).__init__()

        self.gc1 = GraphConvolution(nhid, nembed)
        self.classifier = GraphConvolution(nembed, nclass)
        self.fakereal = GraphConvolution(nembed, 2)
        self.dropout = dropout



    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x_class = F.log_softmax(F.elu(self.classifier(x, adj)), dim=1)  
        x_fakereal = F.log_softmax(F.elu(self.fakereal(x, adj)), dim=1)  

        return self.classifier(x,adj), x_class, x_fakereal, F.softmax(self.classifier(x,adj),dim=1)

class Sage_Discriminator(nn.Module):

    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Discriminator, self).__init__()

        self.gc1 = SageConv(nembed, nhid)
        self.classifier = SageConv(nhid, nclass)
        self.fakereal = SageConv(nhid, 2)
        self.dropout = dropout



    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x_class = F.log_softmax(F.elu(self.classifier(x, adj)), dim=1)  
        x_fakereal = F.log_softmax(F.elu(self.fakereal(x, adj)), dim=1)  

        return self.classifier(x,adj), x_class, x_fakereal, F.softmax(self.classifier(x,adj),dim=1)

class Generator(nn.Module):

    def __init__(self, label_onehot_dim, noise_dim, nembed, dropout):
        super(Generator, self).__init__()
        #self.mlp1 = nn.Linear(noise_dim+label_onehot_dim,200)
        self.mlp1 = nn.Linear(2*label_onehot_dim, 200)   #200 不用mlp2
        #self.mlp1 = nn.Linear(label_onehot_dim, 200)
        self.mlp2 = nn.Linear(800,400)
        self.mlp3 = nn.Linear(200,nembed)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight, std=0.05)
        nn.init.normal_(self.mlp2.weight, std=0.05)
        nn.init.normal_(self.mlp3.weight, std=0.05)
    def forward(self,label_code):
        x = self.mlp1(label_code)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        """
        x = self.mlp2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        """
        x = self.mlp3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        return x
