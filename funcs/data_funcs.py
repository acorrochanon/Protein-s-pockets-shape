import os
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from unisplit import *
# from funcs.unisplit import *

import e3nn
from e3nn import o3
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix


# In order to rotate we create a 3x3 rotation matrix and perform a matrix multiplication on the tensor 
def applyRotation(cavities) -> list:
    # Obtain a random 3x3 matrix
    rand_matrix = o3.rand_matrix(1)
    # print('Random matrix:', rand_matrix)
    
    # Multiply cavity coordinates by the random matrix
    rotated_set = [torch.matmul(cav, rand_matrix).squeeze(0) for cav in cavities]
    
    return rotated_set


# Dataset construction. Decide whether using node features or not (atom types + other scalar features)
def buildDataset(cavities, labels = None, atypes = None, features = None, rotation = True):
    # Load dictionary of atom types
    data_folder  = '/'.join([p for p in os.path.abspath(os.getcwd()).split('/')[:-1]])+'/data'
    #data_folder  = '/'.join([p for p in os.path.abspath(os.getcwd()).split('/')[:-1]])+'/kpjt213/data'
    dict_atypes = torch.load(data_folder+'/Features/dict_atypes.pt')

    if rotation is True:
        cavities = applyRotation(cavities)

    # CAVITIES/ CAVITIES + LABELS
    if atypes is None:
        nodes = [Data(pos = cav) for cav in cavities]

        # Both atom types and features must be used 
        if features is not None: 
                print('ERROR: Atom types and features must be introduced')
                exit(1)

    # CAVITIES + LABELS + NODE FEATURES
    else:  
        nodes = [Data(
                        pos = cav, 
                        x = torch.tensor(np.array([np.append(features[idx],dict_atypes[k]) for k in atypes[idx]]),
                                         dtype=torch.get_default_dtype()))
                   for idx, cav in enumerate(cavities)]

    # LABELS CONDITIONAL
    if labels is None:
        return nodes
    else:
        dataset = [[nodes[i], labels[i]] for i in range(len(labels))]

    return dataset


# Create the DataLoader. Default batch size is 32. 
def makeLoader(dataset, batch = 32, shuf = False):
    
    # Return batch. Shuffle is set to False (default). 
    dataloader = DataLoader(dataset, batch_size = batch, shuffle = shuf)
    
    return dataloader


# splits based on uniprotIDs
def train_test_split(pids, cavities, labels, atypes, features, split = 0.8):
    train_cavities, train_labels, train_atypes, train_features = cavities.copy(), labels.tolist().copy(), atypes.copy(), features.copy()
    test_cavities, test_labels, test_atypes, test_features= [],[],[],[]
    
    # Splitting based on UniprotIDs
    clusters = cluster_protein_structures(pids)
    train_pids, test_pids = uni_split_data(clusters, split)

    for pid in reversed(pids):
        if pid in test_pids:
            test_cavities.append(train_cavities.pop())
            test_labels.append(train_labels.pop())
            test_atypes.append(train_atypes.pop())
            test_features.append(train_features.pop())
        else:
            continue

    #assert len(test_cavities) + len(train_cavities) == len(pids) # Checking that train and test splits equal the og. size

    return (train_cavities, test_cavities), (train_labels, test_labels), (train_atypes, test_atypes), (train_features, test_features)


#Returns the input/output dimension of a gate.
def gate_shape(gate, mode):
    shape = 0
    if mode == 'in':
        gate_type = gate.irreps_in
    else:
        gate_type = gate.irreps_out
        
    for i in gate_type:
        if i[1][0] == 0:
            shape += i[0]
        else:
            shape += (i[1][0]*3)*i[0]
            
    return shape


def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [(x/(float(len(dataset)-1)))**0.5 for x in stdevs]
    return stdevs

# Standardize a list of lists dataset 
def standardize_dataset(dataset):
    means = column_means(dataset)
    stdevs = column_stdevs(dataset, means)
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
    return dataset

# Plot confusion Matrix
def plot_confusion_matrix(cf_matrix, mode = 'train'):
    sns.set(font_scale=1.5)
    plt.figure(figsize = (12, 8))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens', fmt='g')

    ax.set_title(f'Confusion Matrix in {mode} set \n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()