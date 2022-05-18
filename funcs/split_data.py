import os
import sys
from data_funcs import *

current_folder = os.path.dirname(os.path.abspath(__file__))
abs_path = '/'.join([p for p in current_folder.split('/')[:-1]])+'/'


# ARGUMENTS: TRAINING SET SIZE (%, FLOAT)
if __name__ == '__main__':
	path_order = torch.load(abs_path + 'data/Cavities/path_order.pt')
	cavities = torch.load(abs_path +  'data/Cavities/cavities.pt')
	labels = torch.load(abs_path +  'data/Labels/labels.pt')
	atypes = torch.load(abs_path +  'data/Features/atypes.pt')
	features = torch.load(abs_path +  'data/Features/features.pt')

	# TRAIN - TEST SPLITTING 
	cavity_splits, label_splits, atypes_splits, features_splits = train_test_split(
                                                        list(set([i.split('/')[-3].split('_')[0] for i in path_order])),
                                                        cavities, # Filtered cavities
                                                        labels, # Filtered labels
                                                        atypes, # Clean and filtered atom types
                                                        features, # Clean and filtered fpocket features. 
                                                        split = float(sys.argv[1]))

	
	#Cavities
	torch.save(cavity_splits[0], abs_path + 'data/Cavities/train_cavities.pt')
	torch.save(cavity_splits[1], abs_path + 'data/Cavities/test_cavities.pt')

	#Labels
	torch.save(label_splits[0], abs_path + 'data/Labels/train_labels.pt')
	torch.save(label_splits[1], abs_path + 'data/Labels/test_labels.pt')

	# Features - Atom types
	torch.save(atypes_splits[0],abs_path + 'data/Features/train_atypes.pt')
	torch.save(atypes_splits[1],abs_path + 'data/Features/test_atypes.pt')

	# Features - Fpocket features. 
	torch.save(np.array(features_splits[0], dtype = np.float32), abs_path + 'data/Features/train_features.pt')
	torch.save(np.array(features_splits[1], dtype = np.float32), abs_path + 'data/Features/test_features.pt')

	print('Data has been split and saved correctly.')