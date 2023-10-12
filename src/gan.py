import copy
import os
import argparse

import numpy
import numpy as np
import torch.nn as nn
import torch
import torch.distributions.multivariate_normal as mn
import torch.distributions.lowrank_multivariate_normal as lowmn
import torch.distributions.mixture_same_family
import scipy.sparse as sp
from nets import create_gcn, create_gat, create_sage
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import utils
from utils import sample_from_the_distribution
import torch.nn.functional as F

import random
import nni
import data_loads
import models
import torch.optim as optim
import time

import warnings
warnings.filterwarnings("ignore")

from imb_loss import IMB_LOSS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(torch.cuda.is_available())
print(device)

max_recall = 0
f1_best = 0
test_recall = 0
test_f1 = 0
test_AUC = 0
test_acc=0
test_pre =0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer','pubmed','BlogCatalog', 'wiki-cs'])
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--num_im_class', type=int, default=3, choices=[3, 3, 1, 14, 5 ])

    parser.add_argument('--model', type=str, default='gcn', choices=['sage', 'gcn', 'gat'])  
    parser.add_argument('--mode', type=str, default='discrete_edge', choices=['discrete_edge', 'continuous_edge'])
    parser.add_argument('--nhead', type=int, default=8)

    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--nembed', type=int, default=64)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--balance_ratio', type=float, default=0.5)

    parser.add_argument('--loss_type', type=str, default=None, help='loss type', choices=['focal', 'cb-softmax', None])
    parser.add_argument('--factor-focal', default=2.0, type=float, help="alpha in Focal Loss")
    parser.add_argument('--factor-cb', default=0.9999, type=float, help="beta  in CB Loss")


    parser.add_argument('--noise_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr_embed', type=float, default=0.01)
    parser.add_argument('--weight_decay_embed', type=float, default=3e-5)
    parser.add_argument('--lr_gan', type=float, default=0.01)
    parser.add_argument('--weight_decay_gan', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='../log')
    parser.add_argument('--threshold',type=float,default=0.6)

    parser.add_argument('--epoch_gen', type=int, default=3, help='The epoches of generator')   #3
    parser.add_argument('--epoch_dis', type=int, default=500, help='The epoches of discriminator')
    parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)



    if param['dataset'] == 'citeseer':
        param['num_im_class'] = 3

    elif param['dataset'] == 'pubmed':
        param['num_im_class'] = 1

    elif param['dataset'] == 'BlogCatalog':
        param['num_im_class'] = 14
        param['epochs'] = 4010
    elif param['dataset'] == 'wiki-cs':
        param['num_im_class'] = 3
        param['dropout'] = 0.5
        param['lr_embed'] = 0.005
        param['lr_gan']=0.005
        param['dropout']=0.2

    # Load Dataset
    if param['dataset'] == 'cora':
        data, adj, labels, idx_train, idx_val, idx_test, num_classes, num_per_class_list, train_nodes = data_loads.load_cora(num_per_class=20,
                                                                                  num_im_class=param[
                                                                                      'num_im_class'],
                                                                                  im_ratio=param['im_ratio'])
    elif param['dataset'] == 'citeseer':
        data, adj, labels, idx_train, idx_val, idx_test, num_classes, num_per_class_list, train_nodes = data_loads.load_citeseer(num_per_class=20, num_im_class=param['num_im_class'], im_ratio=param['im_ratio'])
    elif param['dataset'] == 'pubmed':
        data, adj, labels, idx_train, idx_val, idx_test, num_classes, num_per_class_list, train_nodes = data_loads.load_pubmed(num_per_class=20,
                                                                                      num_im_class=param[
                                                                                          'num_im_class'],
                                                                                      im_ratio=param['im_ratio'])
    elif param['dataset'] == 'BlogCatalog':
        data, adj, labels, idx_train, idx_val, idx_test, num_classes, num_per_class_list, train_nodes = data_loads.load_BlogCatalog()
    elif param['dataset'] == 'wiki-cs':
        data, adj, labels, idx_train, idx_val, idx_test, num_classes, num_per_class_list, train_nodes = data_loads.load_wiki_cs()
    else:
        print("no this dataset: {param['dataset']}")


    args.log_dir_file = os.path.join(args.log_dir, args.dataset,"gan/")
    os.makedirs(args.log_dir_file, exist_ok=True)

    data = data.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    generator = models.Generator(label_onehot_dim=args.nembed, noise_dim=args.noise_dim, nembed=args.nembed, dropout=args.dropout)
    if args.model == 'sage':
        encoder = create_sage(nfeat=data.x.shape[1], nhid=args.nhid, nclass=num_classes, dropout=args.dropout,
                             nlayer=3, nembed=args.nembed).to(device)
        discriminator = create_sage(nfeat=data.x.shape[1], nhid=args.nhid, nclass=num_classes, dropout=args.dropout,
                                   nlayer=4, nembed=args.nembed).to(device)
    elif args.model == 'gat':
        encoder = create_gat(nfeat=data.x.shape[1], nhid=args.nhid, nclass=num_classes, dropout=args.dropout,
                             nlayer=3, nembed=args.nembed).to(device)
        discriminator = create_gat(nfeat=data.x.shape[1], nhid=args.nhid, nclass=num_classes, dropout=args.dropout,
                                nlayer=4, nembed=args.nembed).to(device)
    elif args.model == 'gcn':
        encoder = create_gcn(nfeat=data.x.shape[1], nhid=args.nhid, nclass=num_classes, dropout=args.dropout,
                             nlayer=3, nembed=args.nembed).to(device)
        discriminator = create_gcn(nfeat=data.x.shape[1], nhid=args.nhid, nclass=num_classes, dropout=args.dropout,
                                   nlayer=4, nembed=args.nembed).to(device)


    optimizer_en = optim.Adam(encoder.parameters(), lr=args.lr_embed, weight_decay=args.weight_decay_embed)
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_gan, weight_decay=args.weight_decay_gan)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_gan, weight_decay=args.weight_decay_gan)


    generator = generator.to(device)
    discriminator = discriminator.to(device)

    best = 0
    best2 = 1000001
    bad_counter = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        encoder.train()
        discriminator.train()
        optimizer_en.zero_grad()
        optimizer_d.zero_grad()
        embed = encoder(data.x, data.edge_index)
        output,_,_,_ = discriminator(embed, data.edge_index)
        loss_train = F.cross_entropy(output[idx_train], data.y[idx_train])
        recall, acc_train, auc_score, macro_F, precision, perclass_values = utils.evaluation(output[idx_train], data.y[idx_train])
        loss_train.backward()
        optimizer_en.step()
        optimizer_d.step()

        if not args.fastmode:
            encoder.eval()
            discriminator.eval()
            embed = encoder(data.x.detach(), data.edge_index.detach())
            output,_,_ ,_= discriminator(embed, data.edge_index)

        loss_val = F.cross_entropy(output[idx_val], data.y[idx_val])
        recall, acc_val, auc_score, macro_F_val, precision, perclass_values = utils.evaluation(output[idx_val], data.y[idx_val])

        if macro_F_val > best:
            best = macro_F_val
            best_epoch = epoch
            bad_counter = 0

        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    embed_class = embed[idx_train]
    labels_class = data.y[idx_train]


    Z = []
    with torch.no_grad():
        for i in range(num_classes):
            Z.append(torch.zeros((1, args.nembed), dtype=torch.float))  # Z: [class_num, nembed]==[class_num, nhid]
        Z = batch2one(Z, labels_class, embed_class)
        N = []
        D = []


        for i in range(num_classes):
            label_embed = Z[i][1:]
            label_mean = torch.mean(label_embed, dim=0).type(torch.float)  # (128,)
            label_var = torch.var(label_embed, dim=0).type(torch.float)
            label_diag = torch.diag(label_var)
            m = mn.MultivariateNormal(loc=label_mean, covariance_matrix=label_diag)
            N.append(label_mean)
            D.append(m)



    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    filename = str(args.log_dir_file+cur_time+"log.txt")


    my_loss = IMB_LOSS(loss_name=args.loss_type, args=args, num_classes=num_classes, c_train_num=num_per_class_list)

    for j in range(args.epoch_dis):
        # Update discriminator
        discriminator.train()
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        latent_code, idx_generated, idx_generated_list, idx_train_new, labels_new = latent_vector_sample(args,
                                                                                                         N, D,
                                                                                                         adj,
                                                                                                         data.y,
                                                                                                         idx_train,
                                                                                                         num_per_class_list=num_per_class_list,
                                                                                                         num_per_class=20,
                                                                                                         num_im_class=args.num_im_class,
                                                                                                         im_ratio=args.im_ratio)
        num_real = idx_train.shape[0]
        num_fake = idx_generated.shape[0]
        embed_generate = generator(latent_code)
        embed_new = torch.cat((embed, embed_generate), dim=0)


        adj_new = torch.sigmoid(torch.mm(embed_new, embed_new.transpose(-1, -2)))

        # Obtain threshold binary edges or soft continuous edges
        if param['mode'] == 'discrete_edge':
            adj_new = copy.deepcopy(adj_new.detach())
            adj_new[adj_new < args.threshold] = 0.0
            adj_new[adj_new >= args.threshold] = 1.0

        num_original = adj.shape[0]
        adj_new[:num_original, :][:, :num_original] = adj.detach()

        adj_new = sp.coo_matrix(adj_new.detach())
        indices = np.vstack((adj_new.row, adj_new.col))
        edge_index_new = torch.LongTensor(indices).to(device)

        logits, fakeorreal_logits, output_class_logsoftmax, output_realfake = discriminator(embed_new.detach(),
                                                                                            edge_index_new.detach())

        if args.loss_type == 'focal' or args.loss_type == 'cb-softmax':
            loss_classification_d = my_loss.compute(logits[idx_train_new], labels_new[idx_train_new].long())
            loss_classification_d = torch.mean(loss_classification_d)
        else:
            loss_classification_d = F.nll_loss(output_class_logsoftmax[idx_train_new],
                                               labels_new[idx_train_new].long())
        label_fakereal_d = torch.cat(
            (torch.LongTensor(num_real).fill_(1), torch.LongTensor(num_fake).fill_(0))).to(device)

        loss_fakeandreal_d = F.nll_loss(output_realfake[idx_train_new], label_fakereal_d)
        print(
            "loss_classificatin_d:{}  loss_fakeandreal_d:{}".format(loss_classification_d, loss_fakeandreal_d))
        loss_dis = loss_classification_d + loss_fakeandreal_d

        loss_dis.backward()
        optimizer_d.step()
        for i in range(args.epoch_gen):
            # update the generator
            generator.train()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()




            logits, fakeorreal_logits, output_class_logsoftmax, output_realfake = discriminator(embed_new.detach(),
                                                                                                edge_index_new.detach())

            label_fakereal_g = torch.LongTensor(num_fake).fill_(1).to(device)
            if args.loss_type == 'focal' or args.loss_type == 'cb-softmax':
                loss_classification_g = my_loss.compute(logits[idx_generated], labels_new[idx_generated].long())
                loss_classification_g = torch.mean(loss_classification_g)
            else:
                loss_classification_g = F.nll_loss(output_class_logsoftmax[idx_generated],
                                                   labels_new[idx_generated].long())

            loss_fakeandreal_g = F.nll_loss(output_realfake[idx_generated], label_fakereal_g)


            loss_gen = loss_classification_g + 0.1 * loss_fakeandreal_g

            print("loss_classification_g:{} and loss_fakeandreal_g:{}".format(loss_classification_g,
                                                                              0.1 * loss_fakeandreal_g))

            loss_gen.backward()
            optimizer_g.step()

            recall_train, acc_train, auc_train, f1_train, precision_train, perclass_train = utils.evaluation(logits=logits[idx_train],labels=labels[idx_train])   
            print("acc:{}, auc:{}, f1:{}".format(acc_train,auc_train,f1_train))
            recall_train_new, acc_train_new, auc_train_new, f1_train_new, precision_train_new, perclass_train_new = utils.evaluation(logits=logits[idx_train_new], labels=labels_new[idx_train_new])
            print("acc_new:{}, auc_new:{}, f1_new:{}".format(acc_train_new, auc_train_new, f1_train_new))
            print('\033[0;30;46m Epoch: discriminator{:04d} generator:{:04d}, train_recall:{:.4f}, train_acc: {:.4f}, train_AUC:{:.4f}, train_f1:{:.4f}, train_precision:{:.4f}\033[0m'.format(i,j, recall_train_new, acc_train_new, auc_train_new, f1_train_new, precision_train_new))
            filecontent = " Epoch:  generator:{%d}, discriminator:{%d},train_recall:{%f}, train_acc: {%f}, train_AUC:{%f}, train_f1:{%f}, train_precision:{%f}" %(i,j, recall_train_new, acc_train_new, auc_train_new, f1_train_new, precision_train_new)
            with open(filename, "a+") as f:
                f.write(filecontent + '\n')


            loss_val, val_recall, val_acc, val_AUC, val_f1_score, val_precision = validate(args,discriminator, embed_new.detach(), edge_index_new,idx_val,idx_test,labels_new)

            print(
            '\033[0;30;41m Epoch: discriminator{:04d} generator:{:04d}, val_recall:{:.4f}, val_acc: {:.4f}, val_AUC:{:.4f}, val_f1_score:{:.4f}, val_precision:{:.4f}\033[0m'.format(
                i, j, val_recall, val_acc, val_AUC, val_f1_score, val_precision))
            filecontent = " Epoch:  generator:{%d}, discriminator:{%d}, val_recall:{%f}, val_acc: {%f}, val_AUC:{%f}, val_f1_score:{%f}, val_precision:{%f}" % (
            i, j, val_recall, val_acc, val_AUC, val_f1_score, val_precision)
            with open(filename, "a+") as f:
                f.write(filecontent + '\n')

    test(filename)
    record(filename,args)

def is_positive_definite(matrix):
    if not np.allclose(matrix.shape[0],matrix.shape[1]):
        return False
    if not np.allclose(matrix,matrix.T):
        return False
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues>0):
        return True
    else:
        return False


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def visualize(args,embedding,labels,idx_train,idx_val,idx_test):
    plot_type = 0
    if plot_type ==0:
        x = embedding[idx_train].detach().cpu().numpy()
        y = labels[idx_train].cpu()
    elif plot_type == 1:
        x = embedding[idx_val].detach().cpu().numpy()
        y = labels[idx_val].cpu()
    elif plot_type == 2:
        x = embedding[idx_test].detach().cpu().numpy()
        y = labels[idx_test].cpu()
    elif plot_type == 3:
        x = embedding.detach().cpu().numpy()
        y = labels.cpu()
    tsne = TSNE(n_components=2)
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    fig, = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", labels.max().item() + 1),
                    data=df).set(title="data T-SNE projection")
    plt.show()
    path_img = "../img"
    path = os.path.join(path_img, args.dataset)
    os.makedirs(path, exist_ok=True)
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    fig_path = '../img/{}/pic-{}-type-{}.png'.format(args.dataset,cur_time,plot_type)
    fig.figure.savefig(fig_path,dpi=400)


def batch2one(Z, y, z):

    for i in range(y.shape[0]):

        Z[y[i]] = torch.cat((Z[y[i]], z[i].view(1,-1).cpu()), dim=0)
    return Z

def compute_cosine(a,b):
    normalize_a = F.normalize(a,p=2,dim=1).to(device)
    normalize_b = F.normalize(b,p=2,dim=0).to(device)
    cosine = torch.matmul(normalize_a,normalize_b.T)
    return cosine




def latent_vector_sample(args, N, D, adj, labels, idx_train,num_per_class_list, num_per_class=20, num_im_class=3, im_ratio=0.5):
    num_classes = len(set(labels.tolist()))
    num_nodes = adj.shape[0]
    num_per_class_generate_list = []
    idx_generated_list = []

    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset =='pubmed':
        for i in range(labels.max().item() + 1):
            if i > labels.max().item() - num_im_class:
                num_per_class_generate_list.append(int(num_per_class - num_per_class * im_ratio))
            else:
                num_per_class_generate_list.append(num_per_class - num_per_class)
    elif args.dataset == 'wiki-cs':
        mean0 = sum(num_per_class_list)/len(num_per_class_list)
        for i in range(labels.max().item()+1):
            if i > labels.max().item()-num_im_class:
                num_per_class_generate_list.append(int(mean0-num_per_class_list[i]))
            else:
                num_per_class_generate_list.append(0)


    total_generated = 0
    if args.noise == True:
        latent_code = torch.ones(1, args.nembed)
    else:
        latent_code = torch.ones(1,  2*args.nembed)

    for label in range(labels.max().item() + 1):
        label = int(label)
        if label > labels.max().item() - num_im_class:
            num_generate = int(num_per_class_generate_list[label])
            total_generated = total_generated + num_generate
            if args.noise == True:
                conditional_z = torch.FloatTensor(np.random.normal(0, 0.1, (num_generate, args.nembed))).to(device)
            else:
                conditional_z = D[label].sample().view(-1, args.nembed)
                for j in range(num_generate-1):
                    z = D[label].sample().view(-1, args.nembed)
                    conditional_z = torch.cat((conditional_z, z), dim=0)
            label_train_append = label * np.ones(num_generate)

            tmp_code = latent_code.to(device)

            if args.noise == True:
                latent_code = torch.cat((tmp_code, conditional_z),dim=0).to(device)
            else:
                latent_code = torch.cat((conditional_z, N[label].repeat(num_generate, 1)), dim=1).to(device)
                latent_code = torch.cat((tmp_code, latent_code), dim=0).to(device)
            idx_train_append = idx_train.new(np.arange(num_nodes, num_nodes + num_generate))
            idx_train = torch.cat((idx_train, idx_train_append), 0)
            idx_generated_list.append(idx_train_append)
            num_nodes = num_nodes + num_generate
            labels = torch.cat((labels, torch.Tensor(label_train_append).to(device)), 0)
        else:
            idx_generated_list.append(0)

    idx_generated = idx_train[(idx_train.shape[0] - total_generated):]

    # return idx_train_new, labels_new
    return latent_code[1:, :], idx_generated, idx_generated_list, idx_train, labels


def validate(args,discriminator,embed,edge_index,idx_val,idx_test,labels):
    global max_recall, test_recall, test_f1, test_AUC, test_acc, test_pre,f1_best,test_perclass
    if not args.fastmode:
        discriminator.eval()
        logits, fakeorreal_logits, output, output_gen  = discriminator(embed.detach(), edge_index.detach())
        loss_val = F.nll_loss(output[idx_val],labels[idx_val].long())

    recall_val, acc_val, AUC_val, f1_val, pre_val, perclass_val = utils.evaluation(logits[idx_val],labels[idx_val])


    if f1_val > f1_best:
        logits,fakeorreal_logits, output, output_gen = discriminator(embed, edge_index)
        recall_tmp, acc_tmp, AUC_tmp, f1_tmp, pre_tmp,perclass_tmp = utils.evaluation(logits[idx_test],labels[idx_test])
        test_recall = recall_tmp
        test_f1 = f1_tmp
        test_AUC = AUC_tmp
        test_acc = acc_tmp
        test_pre = pre_tmp
        test_perclass = perclass_tmp
        f1_best = f1_val


    return loss_val, recall_val, acc_val, AUC_val, f1_val, pre_val

def test(filename):
    print("Test Recall: ", test_recall)
    print("Test Accuracy: ", test_acc)
    print("Test F1: ", test_f1)
    print("Test precision: ", test_pre)
    print("Test AUC: ", test_AUC)
    print("Test perclass:")
    print(test_perclass)
    filecontent = " test_recall:{%f}, test_acc_score: {%f}, test_AUC:{%f}, test_f1:{%f}, test_precision:{%f}" % (
        test_recall, test_acc, test_AUC, test_f1, test_pre)
    with open(filename, "a+") as f:
        f.write(filecontent + '\n')


def record(filename,args):
    argsDict = args.__dict__
    with open(filename, 'a+') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        f.close


def normalize(mx):
    mx = mx.cpu()
    rowsum = mx.sum(1).detach().numpy()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx.detach())
    return torch.Tensor(mx).to(device)



if __name__ == '__main__':
    main()
