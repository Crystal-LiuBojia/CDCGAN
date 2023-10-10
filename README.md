# CDCGAN
Class Distribution-aware Conditional GAN-based Minority Augmentation for Imbalanced Node Classification
## Requirements
This repository has been tested with the following packages:
+ Python == 3.7.13
+ PyTorch == 1.12.1
+ Pytorch Geometric == 2.3.1
## Important Hyper-parameters
+ `--dataset`: name of the dataset. It could be one of `['cora', 'citeseer','pubmed', 'wiki-cs']`.
+ `im_ratio`: imbalance ratio.
+ `model`: name of the backbone encoder. It could be one of `['sage', 'gcn', 'gat']`.
+ `mode`: the generation way of new edges.
+ `threshold`: the threshold of edge generation.
## How to run
For example:
`python gan.py --dataset cora --im_ratio 0.5 --model gcn --threshold 0.6`



