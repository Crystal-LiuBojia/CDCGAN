import random
import numpy as np
from copy import deepcopy
import sklearn
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, classification_report
from scipy.spatial.distance import pdist, squareform


import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(torch.cuda.is_available())
print(device)

def evaluation(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    accuracy = correct.item() * 1.0 / len(labels)

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1).detach().cpu().numpy(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy(), average='macro')

    recall = recall_score(labels.cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy(), average='macro')
    macro_F = f1_score(labels.detach().cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy(), average='macro')
    precision = precision_score(labels.detach().cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy(),
                       average='macro')
    perclass_values = classification_report(labels.detach().cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy())

    return recall, accuracy, auc_score, macro_F, precision, perclass_values

def accuracy(output, labels, output_AUC):
    preds = output.max(1)[1].type_as(labels)

    recall = sklearn.metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    f1_score = sklearn.metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    AUC = sklearn.metrics.roc_auc_score(labels.cpu().numpy(), output_AUC.detach().cpu().numpy(),multi_class='ovr',average='macro')   
    acc = sklearn.metrics.accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = sklearn.metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    return recall, f1_score, AUC, acc, precision

# Interpolation in the input space
def src_upsample(features, labels, idx_train, adj, up_scale=1.0, im_class_num=3):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    chosen = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / new_chosen.shape[0]) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/new_chosen.shape[0] - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale

        for j in range(c_up_scale):
            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            
        if up_scale_rest != 0:
            num = int(new_chosen.shape[0] * up_scale_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num)))
    new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
    new_adj[adj.shape[0]:, :adj.shape[0]] = adj[chosen,:]
    new_adj[:adj.shape[0], adj.shape[0]:] = adj[:,chosen]
    new_adj[adj.shape[0]:, adj.shape[0]:] = adj[chosen,:][:,chosen]

    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_train_append = idx_train.new(np.arange(adj.shape[0], adj.shape[0] + add_num))    

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return features, labels, idx_train, new_adj.detach()
# Sample from the distribution
def sample_from_the_distribution(args, distribution,embed, labels, idx_train, adj, up_scale=1.0, num_per_class=20, num_im_class=3, im_ratio=0.5):   # im_class_num都是以cora数据集为标准的
    num_classes = len(set(labels.tolist()))
    num_nodes = adj.shape[0]
    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class-num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class-num_per_class)    
    total_generated = 0
    label_onehot = F.one_hot(labels)  
    for label in range(labels.max().item() + 1):
        label = int(label)
        if label > labels.max().item() - num_im_class:
            num_generate = int(num_per_class_list[label])
            total_generated = total_generated + num_generate
            #print("[*]Generating {} label({}) latent vector".format(num_generate, label))
            if args.noise==False:
                conditional_z = distribution[label].sample((num_generate,))
            else:
                conditional_z = torch.FloatTensor(np.random.normal(0, 0.1, (num_generate,args.nembed)))   
            conditional_z = conditional_z.to(device)    
            latent_code = conditional_z.view(-1, args.nembed)  
            embed = torch.cat((embed,latent_code),dim=0)

            idx_train_append = idx_train.new(np.arange(num_nodes, num_nodes + num_generate))
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            num_nodes = num_nodes + num_generate
            label_train_append = label * np.ones(num_generate)
            labels = torch.cat((labels,torch.Tensor(label_train_append).to(device)),0)

    embed_generated = embed[(embed.shape[0]-total_generated):,:]
    idx_generated = idx_train[(idx_train.shape[0] - total_generated):]

    return embed, embed_generated, idx_generated, idx_train, labels



# Interpolation in the embedding space
def src_smote(features, labels, idx_train, adj, up_scale=1.0, im_class_num=3):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    chosen = None
    new_features = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / new_chosen.shape[0]) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/new_chosen.shape[0] - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale
            
        for j in range(c_up_scale):

            chosen_embed = features[new_chosen, :]

            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)
        
        if up_scale_rest != 0.0 and int(new_chosen.shape[0] * up_scale_rest)>=1:

            num = int(new_chosen.shape[0] * up_scale_rest)
            new_chosen = new_chosen[:num]
            chosen_embed = features[new_chosen, :]

            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
                
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num)))
    new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
    new_adj[adj.shape[0]:, :adj.shape[0]] = adj[chosen,:]
    new_adj[:adj.shape[0], adj.shape[0]:] = adj[:,chosen]
    new_adj[adj.shape[0]:, adj.shape[0]:] = adj[chosen,:][:,chosen]

    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_train_append = idx_train.new(np.arange(adj.shape[0], adj.shape[0] + add_num))

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return features, labels, idx_train, new_adj.detach()

# Mixup in the semantic relation space
def mixup(embed, labels, idx_train, adj=None, up_scale=1.0, im_class_num=3, scale=0.0):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / chosen.shape[0] + scale) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/chosen.shape[0] + scale - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
            # print(round(scale, 2), round(c_up_scale, 2), round(up_scale_rest, 2))
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale
            

        for j in range(c_up_scale):

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()

            new_embed = embed[chosen, :] + (embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0]))

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_new), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

        if up_scale_rest != 0.0:

            num = int(chosen.shape[0] * up_scale_rest)
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()

            new_embed = embed[chosen, :] + (embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0]))

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_new), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        if adj_new is not None:
            add_num = adj_new.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

            return embed, labels, idx_train, new_adj.detach()
        else:
            return embed, labels, idx_train, adj.detach()

    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt, param):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2
    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    if param['dataset'] == 'cora':
        return loss * 1e-3
    else:
        return loss / adj_tgt.shape[0]




