import json
import torch
import random
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from torch_geometric.data import Data


# Load Cora Dataset
def load_cora(num_per_class=20, num_im_class=3, im_ratio=0.5):

    print('Loading cora dataset...')
    edges_unordered = np.genfromtxt("../data/cora/cora.cites", dtype=np.int32)
    idx_features_labels = np.genfromtxt("../data/cora/cora.content", dtype=np.dtype(str))


    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]    # 取的是最后一列
    
    #classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}
    classes_dict = {'Neural_Networks': 0, 'Probabilistic_Methods': 1, 'Genetic_Algorithms': 2, 'Theory': 3, 'Case_Based': 4, 'Reinforcement_Learning': 5, 'Rule_Learning': 6}
    # 按照x
    labels = np.array(list(map(classes_dict.get, labels)))    # 把对应的领域编号。labels是一个里面全是0，1，2，3，4，5，6的array数组

    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_old = torch.FloatTensor(np.array(adj.todense()))

    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

    features = torch.FloatTensor(np.array(normalize_features(features).todense()))
    labels = torch.LongTensor(labels)
    # adj = torch.FloatTensor(np.array(adj.todense()))

    data = Data(x=features, edge_index=edge_index, y=labels)

    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class)

    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    train_nodes = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()     # c_idx是对应的某一个类label==i的里面的id号，转换成list（tolist）
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        train_idx = train_idx + c_idx[:num_per_class_list[i]]   # 列表加就是把列表合并，label不用shuffle因为可以根据id进行搜索
        val_idx = val_idx + c_idx[num_per_class_list[i]:num_per_class_list[i]+25]   # 验证集和测试集的数量，每个类都是一样的，验证机25个，测试集80-25=55个
        test_idx = test_idx + c_idx[num_per_class_list[i]+25:num_per_class_list[i]+80]

        train_nodes.append(c_idx[:num_per_class_list[i]])

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    # return train_idx, val_idx, test_idx, adj, features, labels, num_classes
    return data, adj_old, labels, train_idx, val_idx, test_idx, num_classes, num_per_class_list, train_nodes
def load_citeseer(num_per_class=20, num_im_class=3, im_ratio=0.5):
    print('Loading citeseer dataset...')

    idx_features_labels = np.genfromtxt("../data/citeseer/citeseer.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("../data/citeseer/citeseer.cites", dtype=np.int32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    classes_dict = {'Agents':0,'AI':1,'DB':2,'IR':3,'ML':4,'HCI':5}
    #classes_dict = {'DB':0,'IR':1,'Agents':2,'ML':3,'HCI':4,'AI':5}
    # labels = {'Agents':1,'AI':2,'DB':3,'IR':4,'ML':5,'HCI':6}
    labels = np.array(list(map(classes_dict.get, labels)))

    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_old = torch.FloatTensor(np.array(adj.todense()))

    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

    features = torch.FloatTensor(np.array(normalize_features(features).todense()))
    labels = torch.LongTensor(labels)
    # adj = torch.FloatTensor(np.array(adj.todense()))

    data = Data(x=features, edge_index=edge_index, y=labels)
    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class)

    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    train_nodes = []

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        train_idx = train_idx + c_idx[:num_per_class_list[i]]
        val_idx = val_idx + c_idx[num_per_class_list[i]:num_per_class_list[i] + 25]
        test_idx = test_idx + c_idx[num_per_class_list[i] + 25:num_per_class_list[i] + 80]
        train_nodes.append(c_idx[:num_per_class_list[i]])

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    # return train_idx, val_idx, test_idx, adj, features, labels, num_classes
    return data, adj_old, labels, train_idx, val_idx, test_idx, num_classes, num_per_class_list, train_nodes

def load_pubmed(num_per_class=20, num_im_class=1, im_ratio=0.5):
    print('Loading pubmed dataset...')

    idx_features_labels = np.genfromtxt("../data/pubmed/pubmed.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("../data/pubmed/pubmed.cites", dtype=np.int32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]    # 最后一列就是1，2，3

    #classes_dict = {'1':0,'2':1,'3':2}     # 从0开始编号
    classes_dict = {'2':0,'3':1,'1':2}   # 从d
    # labels = {'Agents':1,'AI':2,'DB':3,'IR':4,'ML':5,'HCI':6}
    labels = np.array(list(map(classes_dict.get, labels)))

    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_old = torch.FloatTensor(np.array(adj.todense()))

    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

    features = torch.FloatTensor(np.array(normalize_features(features).todense()))
    labels = torch.LongTensor(labels)
    # adj = torch.FloatTensor(np.array(adj.todense()))

    data = Data(x=features, edge_index=edge_index, y=labels)

    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class)

    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    train_nodes = []
    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        train_idx = train_idx + c_idx[:num_per_class_list[i]]
        val_idx = val_idx + c_idx[num_per_class_list[i]:num_per_class_list[i] + 25]
        test_idx = test_idx + c_idx[num_per_class_list[i] + 25:num_per_class_list[i] + 80]
        train_nodes.append(c_idx[:num_per_class_list[i]])

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    # return train_idx, val_idx, test_idx, adj, features, labels, num_classes
    return data, adj_old, labels, train_idx, val_idx, test_idx, num_classes, num_per_class_list, train_nodes

# Load BlogCatalog Dataset
def load_BlogCatalog():
    mat = loadmat('./data/BlogCatalog/blogcatalog.mat')
    embed = np.loadtxt('./data/BlogCatalog/blogcatalog.embeddings_64')

    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]
    features = normalize_features(feature)

    adj = mat['network']
    label = mat['group']

    labels = np.array(label.todense().argmax(axis=1)).squeeze()
    labels[labels>16] = labels[labels>16]-1
    labels = refine_label_order(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(np.array(adj.todense()))


    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        if c_num < 4:
            if c_num < 3:
                print("too small class type")
            batch_train = 1
            batch_val = 1
            batch_test = 1
        else:
            batch_train = int(c_num/4)
            batch_val = int(c_num/4)
            batch_test = int(c_num/2)

        train_idx = train_idx + c_idx[:batch_train]
        val_idx = val_idx + c_idx[batch_train:batch_train+batch_val]
        test_idx = test_idx + c_idx[batch_train+batch_val:batch_train+batch_val+batch_test]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, adj, features, labels, num_classes


# Load Wiki-CS Dataset
def load_wiki_cs():

    raw = json.load(open('../data/wiki-cs/data.json'))
    features = torch.FloatTensor(np.array(raw['features']))
    labels = np.array(raw['labels'])

    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(raw['links'])]))
    src, dst = tuple(zip(*edge_list))
    adj = np.unique(np.array([src, dst]).T, axis=0)
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_old = torch.FloatTensor(np.array(adj.todense()))

    adj = torch.FloatTensor(np.array(adj.todense()))
    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

    # features = torch.FloatTensor(np.array(normalize_features(features).todense()))
    # labels = torch.LongTensor(labels)
    # adj = torch.FloatTensor(np.array(adj.todense()))



    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_list = []
    classes_dict = {4: 0, 2: 1, 3: 2, 9: 3, 7: 4, 5: 5, 1: 6, 8: 7, 6: 8, 0: 9}  # 按照从多到少的顺序排序
    #labels = {'Agents':1,'AI':2,'DB':3,'IR':4,'ML':5,'HCI':6}
    labels = np.array(list(map(classes_dict.get, labels)))
    labels = torch.LongTensor(labels)
    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        c_num_list.append(c_num)
    c_num_list.sort(reverse=True)
    print(c_num_list)
    data = Data(x=features, edge_index=edge_index, y=labels)

    train_nodes = []
    num_per_class_list = []
    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        #print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        if c_num < 4:
            if c_num < 3:
                print("too small class type")
            batch_train = 1
            batch_val = 1
            batch_test = 1
        else:
            batch_train = int(c_num/4)
            batch_val = int(c_num/4)
            batch_test = int(c_num/2)

        train_idx = train_idx + c_idx[:batch_train]
        val_idx = val_idx + c_idx[batch_train:batch_train+batch_val]
        test_idx = test_idx + c_idx[batch_train+batch_val:batch_train+batch_val+batch_test]
        train_nodes.append(c_idx[:batch_train])
        num_per_class_list.append(batch_train)

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    # return train_idx, val_idx, test_idx, adj, features, labels, num_classes

    return data, adj_old, labels, train_idx, val_idx, test_idx, num_classes, num_per_class_list, train_nodes
    """
    raw = json.load(open('../data/wiki-cs/data.json'))
    features = torch.FloatTensor(np.array(raw['features']))
    labels = torch.LongTensor(np.array(raw['labels']))

    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(raw['links'])]))
    src, dst = tuple(zip(*edge_list))
    adj = np.unique(np.array([src, dst]).T, axis=0)
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max() + 1, adj.max() + 1),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_old = torch.FloatTensor(np.array(adj.todense()))

    adj = torch.FloatTensor(np.array(adj.todense()))
    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

    # features = torch.FloatTensor(np.array(normalize_features(features).todense()))
    # labels = torch.LongTensor(labels)
    # adj = torch.FloatTensor(np.array(adj.todense()))

    data = Data(x=features, edge_index=edge_index, y=labels)

    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    train_nodes = []
    num_per_class_list = []
    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        if c_num < 4:
            if c_num < 3:
                print("too small class type")
            batch_train = 1
            batch_val = 1
            batch_test = 1
        else:
            batch_train = int(c_num / 4)
            batch_val = int(c_num / 4)
            batch_test = int(c_num / 2)

        train_idx = train_idx + c_idx[:batch_train]
        val_idx = val_idx + c_idx[batch_train:batch_train + batch_val]
        test_idx = test_idx + c_idx[batch_train + batch_val:batch_train + batch_val + batch_test]
        train_nodes.append(c_idx[:batch_train])
        num_per_class_list.append(batch_train)

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    # return train_idx, val_idx, test_idx, adj, features, labels, num_classes
    return data, adj_old, labels, train_idx, val_idx, test_idx, num_classes, num_per_class_list, train_nodes
    """
def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(), 0, -1):
        if sum(labels==i) >= 101 and i > j:
            while sum(labels==j) >= 101 and i > j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j + 1
            else:
                break
        elif i <= j:
            break

    return labels

"""
def normalize(mx):
    # Row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
"""
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx