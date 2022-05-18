import os
import sys
import time
import pandas as pd
from importlib import reload

current_folder = os.path.dirname(os.path.abspath(__file__))
abs_path = '/'.join([p for p in current_folder.split('/')[:-1]])+'/'
sys.path.insert(0, abs_path + 'funcs')

# Manual files
import scpdb_funcs
from scpdb_funcs import *
import fpocket_funcs
from fpocket_funcs import *
import common_funcs
from common_funcs import *

# Reload if modified. 
reload(scpdb_funcs)
reload(fpocket_funcs)
reload(common_funcs)


# ARGUMENTS: NUMBER OF PATHS (Integer)
if __name__ == '__main__':
	# READ ALL THE SCPDB PATHS CONTAINING THE CAVITIES
	paths = [x[0] for x in os.walk(abs_path+'scPDB')][1:]

	# NUMBER OF PATHS THAT WILL BE USED
	NUM_PATHS = len(paths)
	
	# PDB FOLDER WHERE ALL THE FPOCKET CAVITIES WILL BE CONTAINED 
	folder_pdb_name = 'example'

	# FPOCKET PARAMETERS
	best_fp_params = '-m 3.5 -M 6 -i 30 -r 4.5 -s 2.5'
	# -m 3.5 -M 6 -i 30 -r 4.3 -s 2.3

	# GET sc-PDB DATA (CAVITY, VOLUMES..)
	print('SC-PDB HUNTING...')
	start_time = time.time()
	sc_cavities, points_per_pocket_sc, pdb_dict, sc_volumes = getScData(paths, NUM_PATHS, folder_pdb_name)
	print(f'Execution time: {(time.time() - start_time)/60:.2f} minutes\n')

	# RUN FPOCKET 
	start_time = time.time()
	getFpockets(folder_pdb_name, best_fp_params)
	print(f'Execution time: {(time.time() - start_time)/60:.2f} minutes\n')


	# EXTRACT FPOCKET POCKET CAVITIES, ATOM TYPES, FEATURES. SAVE READING ORDER.
	fp_coords, fp_atypes, fp_info, path_order, pdb_order = generatePocketStructures(folder_pdb_name, pdb_dict)
	# STATS: Number of pockets per protein structure
	pockets_per_struct = [len(fp_info[idx]) for idx in range(len(fp_info))]
	# STAS: Number of points per pocket
	points_per_pocket_fp = [len(pocket_fp) for protein_fp in fp_coords for pocket_fp in protein_fp]
	
	print(f'{len(fp_coords)} druggable cavities')
	print(f'{len(fp_info[0][0])} features in each pocket')
	print(f'Total number of pockets: {sum(pockets_per_struct)}\n')


	# GET LABELS OF DRUGGABLE FPOCKET POCKETS AND THEIR RESPECTIVE DISTANCES TO SCPDB POCKET CENTROID
	labels, distances = getLabels(fp_coords, sc_cavities, pdb_order)
	print(f'Mean distance: {stats.mean(distances):.2f}')
	print(f'Median distance: {stats.median(distances):.2f}\n')

	# GET PROTEIN STRUCTURES IDS AND INDICES 
	pid_list, druggable_indices = findDrugPDBids(distances, labels, path_order)

	# FILTERED CAVITIES BASED ON ESTABLISHED REQUIREMENTS 
	filt_cavities, filt_labels, filt_atypes, fp_info_filt, filt_path_order = filterDataset(flatList(fp_coords), flatList(fp_atypes), 
	                                         labels, distances, pid_list, druggable_indices, path_order, flatList(fp_info))

	# ATOM TYPES CLEANING (SOME CONTAIN APOSTROPHES)
	for idx1, pocket in enumerate(filt_atypes):
	    for idx2, atom_type in enumerate(pocket):
	        filt_atypes[idx1][idx2] = ''.join(elem for elem in atom_type if elem.isalnum())
	        
	# Generate dictionary of the unique atom types we are working with.
	dict_atypes = atypesToDict(filt_atypes)

	# COMPUTE VOLUME FOR FILTERED CAVITIES.
	vols = [getPocketVolume(cav) for cav in filt_cavities]

	# REMOVING NON-DESIRED FPOCKET FEATURES: POCKET-SCORE, MEAN BFACTOR, VOLUME (REPLACED)
	for idx, pocket in enumerate(fp_info_filt):
	    for pos in [8, 4, 0]: # AZ: 10, 9, 4, 1, 0
	        del(pocket[pos])
	    pocket.append(vols[idx])


	# DRUGGABLE AND NON DRUGGABLE PERCENTAGES
	drug = len(np.where(filt_labels>0)[0])
	non_drug = len(np.where(filt_labels==0)[0])

	print(f"druggable cavities: {(drug/len(filt_labels))*100:.2f}% ({drug})")
	print(f"non druggable cavities: {(non_drug/len(filt_labels))*100:.2f}% ({non_drug})")

	#SAVE THE DATA
	torch.save(filt_cavities, abs_path + 'data/Cavities/cavities.pt')
	torch.save(filt_labels, abs_path + 'data/Labels/labels.pt')
	torch.save(filt_atypes, abs_path + 'data/Features/atypes.pt')
	torch.save(fp_info_filt, abs_path + 'data/Features/features.pt')
	torch.save(dict_atypes, abs_path + 'data/Features/dict_atypes.pt')
	torch.save(filt_path_order, abs_path + 'data/Cavities/path_order.pt')

