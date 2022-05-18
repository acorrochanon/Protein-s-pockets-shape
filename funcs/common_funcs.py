# FILE CONTAINS FUNCTIONS THAT ARE CONSIDERED TO NOT BE PART OF TWO MAIN PIPELINES (SCPDB AND FPOCKET)

import os 
import torch
import numpy as np
import seaborn as sns
from copy import deepcopy 
import scipy.spatial as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics as stats


# Returns centroid from a given point cloud. Unsqueeze (increase dimensionality) required 
# to compute distance later. Receives a single tensor. 
def getCentroid(point_cloud) -> torch.Tensor:
    return torch.mean(point_cloud, dim = 0).unsqueeze(0)


# Compute euclidean distance between two tensor-type centroids
def calculateDistance(t1, t2) -> float:
    return torch.cdist(t1, t2, p=2).item()


# Flat list of lists
def flatList(list_of_lists) -> list:
    if type(list_of_lists[0][0]) == torch.Tensor and len(list_of_lists[0][0]) == 1:
        return [item.item() for sublist in list_of_lists for item in sublist]
    
    else:
        return [item for sublist in list_of_lists for item in sublist]

# Clean atom types list, as some of them may contain apostrophes
def cleanAtypes(atypes):
    clean_atypes = [''.join(e for e in atom_type if e.isalnum()) for atom_type in flatList(atypes)]
    return clean_atypes


# Receives a list of lists (e.g, all the atom types a protein contains). Returns dict 
def atypesToDict(atypes) -> dict:
    # Get the unique set of features. 
    unique_atypes = set(flatList(atypes))

    # Generate dictionary for posterior mapping/encoding of features.
    dict_atypes = {element:idx for idx, element in enumerate(unique_atypes)}
    
    return dict_atypes


# Generate .txt filled with the paths where the .pdb pocket files are located.
# Fpocket will be executed against this fileList  
def createTxt(data, folder_name):
    file_name = 'pdbList'
    textfile = open(file_name, "w")
    for idx, element in enumerate(data):
        textfile.write(element+'\n')
    textfile.close()
    
    return file_name


# Obtain final set of labels. 
def getLabels(fp_coords, sc_cavities, pdb_order, pdb_dict):
    labels, distances = [], []

    for idx, (fp_protein, order) in enumerate(zip(fp_coords, pdb_order)):
        # Calculate centroid for each point cloud 
        fp_centroids = [getCentroid(pocket) for pocket in fp_protein]

        # Calculate the distance between centoids (fpocket's centroid vs scpdb's)
        # We need to take into account the order in which proteins were read, so we can calculate the distance 
        # with the scPDB cavity of the same structure. 
        p_distances = [calculateDistance(fpcent, getCentroid(sc_cavities[order])) for fpcent in fp_centroids]
        # In some cases we dont have pockets. We print the ID to remove it in the next run 
        if len(p_distances) == 0:
            print(check_pdbid(order, pdb_dict))
            continue

        # Generate labels. All 0 except for the pocket that holds the closest distance to the scPDB's centroid.
        p_labels = torch.zeros(len(fp_centroids), 1)
        
        # The closest distance will be stored for each pair of centroids for later visualization. 
        distances.append(min(p_distances))
        p_labels[p_distances.index(min(p_distances))] = 1        
        
        # Append to overall list (protein-level)
        labels.append(p_labels)
    
    # Return pocket-level label list and minimum distances for every fpocket's closest pocket to scpdb's
    return flatList(labels), distances 


# Returns the IDs of the pockets labeled as druggable. 
def findDrugPDBids(distances, labels, path_order):
    # Find indices where druggable cavities are located
    drug_idx = np.where(np.array(labels) > 0)[0]
    pocket_id_list = []
    for idx, i in enumerate(drug_idx):
        # Grab ID of fpocket cavity selected as druggable 
        pocket_id = int(path_order[i].split('/')[-1].split('_')[0].replace('pocket',''))
        pocket_id_list.append(pocket_id)
        
        # Print the path of the druggable pocket, its distance to scpdb's protein pocket.
        # print(idx, path_order[i].split('/')[-3:], round(distances[idx], 2),'Å,',

    # Countplot visualization
    sns.set_theme()
    plt.figure(figsize = (20, 5))
    ax = sns.countplot(x = pocket_id_list)
    for p in ax.patches:
        count = p.get_height()
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(count, (x, y), size = 16)
    plt.xlabel('Pocket ID', fontsize = 18)
    plt.xticks(fontsize=14)
    plt.ylabel('Count', fontsize = 18)
    plt.yticks(fontsize=14)
    plt.title('IDs of the Fpocket cavities labeled as druggable', fontsize = 18)
    plt.savefig('pocketIDs', dpi = 300)
    
    return pocket_id_list, drug_idx


# Filter the data with only those cavities with the greatest pocket score and smallest distance 
def filterDataset(pocket_coords, temp_atypes, temp_labels, distances, fpids, drug_idx, path_order, fp_info):
    # List will contain all the ids that need to be removed
    remove_inds = []
    
    # Create temporary structures to not delete elements in main ones. 
    copy_pockets = deepcopy(pocket_coords)
    copy_atypes = deepcopy(temp_atypes)
    copy_labels = deepcopy(temp_labels)
    copy_path = deepcopy(path_order)
    # copy_fpinfo = deepcopy(fp_info)
    
    # Distance less than 5 and pocket id 0/1.
    for idx, (distance, fpid) in enumerate(zip(distances, fpids)):
        if distance > 5 or fpid > 1:
            remove_inds.append(drug_idx[idx])
    
    # Number of points condition to balance the dataset
    for idx, pocket in enumerate(copy_pockets):
        if len(pocket) < 30:
            # Check if the pocket has not been yet added to the removal list. 
            if idx not in remove_inds:
                remove_inds.append(idx)

    # Remove coordinates, labels, and atom types of the coordinates 
    for rem_idx in sorted(remove_inds, reverse = True):
        del(copy_pockets[rem_idx])
        del(copy_atypes[rem_idx])
        del(copy_path[rem_idx])
        del(fp_info[rem_idx])

    # Keep the labels of the cavities that were not deleted. 
    copy_labels = np.delete(np.array(temp_labels), remove_inds)
    
    print(f'Removing {len(remove_inds)} cavities')
    print(f'Final number of pockets:{len(copy_pockets)}\n')
    
    # Return filtered structures
    return copy_pockets, copy_labels, copy_atypes, fp_info, copy_path


# Visualize pockets convex hull
def visualizePocket(cavity):
    pts = cavity.numpy()
    hull = ss.ConvexHull(pts)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")

    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))
    
    print('Volume estimation (ConvexHull): {:.2f}'.format(hull.volume))
    print('Number of points: {}'.format(len(pts)))
    
    ax.set_xlabel('X', fontsize = 20)
    ax.set_ylabel('Y', fontsize = 20)
    ax.set_zlabel('Z', fontsize = 20)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    plt.savefig('Images/cavfig', bbox_inches = 'tight', dpi = 300)
    plt.show()
    
    
# Return pdb id based on index
def check_pdbid(index, pdb_dict):
    for pos, i in enumerate(pdb_dict.values()):
        if index in i:
            return list(pdb_dict.keys())[pos]