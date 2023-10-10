# CDCGAN
Class Distribution-aware Conditional GAN-based Minority Augmentation for Imbalanced Node Classification
## DATASETS
We adopt four widely used datasets in the node classification task, including Cora, CiteSeer, PubMed, and Wiki-CS. The details of these four datasets are listed as follows:

+ Cora: The Cora dataset is an academic citation network that models the referential relations among scientific papers. In this network, nodes represent different papers, and edges represent mutual citation relations. At the same time, each node in the Cora is associated with a 0/1-valued attribute vector describing the absence or presence of a certain word in the dictionary. This dataset contains 2,708 nodes with 1,433-dimensional features and 5,429 citations links. In addition, labels of nodes indicate the seven research fields the publications belong to.

+ CiteSeer: The CiteSeer dataset is an academic dataset with 3,327 nodes from six classes and 4,732 edges. Similar to Cora, each publication in the dataset is described with a 3,703-dimensional attribute feature valued at 0 and 1, indicating the absence and presence of these words. 

+ PubMed: The PubMed dataset is an academic network related to diabetes with 19,717 nodes, 44,338 citation edges, and three categories. The attribute information of nodes is described by a TF/IDF weighted word from the dictionary. The most minor type of publication has only 4,103 samples, resulting in the dataset imbalance.
+ Wiki-CS: It is a novel dataset collected from Wikipedia in the Computer Science domain, especially for semi-supervised node classification tasks of Graph Neural Networks. The dataset has 11,701 nodes corresponding to CS articles of 10 branches and 216,123 edges representing hyperlinks. In addition, node attribute features are calculated as the 300-dim pre-trained GloVe word embeddings rather than simple binary bag-of-words vectors.
