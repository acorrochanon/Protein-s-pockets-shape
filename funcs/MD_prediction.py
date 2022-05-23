import os
import torch
import umap
import numpy as np
import scipy.spatial as ss
from data_funcs import buildDataset, makeLoader
from materials import Network, Convolution
from biopandas.pdb import PandasPdb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns



def getCentroid(point_cloud) -> torch.Tensor:
	return torch.mean(point_cloud, dim = 0).unsqueeze(0)


# Compute euclidean distance between two tensor-type centroids
def calculateDistance(t1, t2) -> float:
    return torch.cdist(t1, t2, p=2).item()

# Calculate the volume of the mesh that each point cloud/cavity conforms. 
def getPocketVolume(cavity):
    return ss.ConvexHull(cavity).volume



if __name__ == '__main__':
	# Construct a UMAP object
	reducer = umap.UMAP()
	# Biopandas reader
	ppdb = PandasPdb()
	# Set fpocket parameters.
	fpocket_params = '-m 3.5 -M 6 -i 30 -r 4.5 -s 2.5'
	# Initialize the model 
	model = Network()
	model.load_state_dict(torch.load('1a_model_bce_4cv.pt', map_location=torch.device('cpu')))
	model.eval()

	# Initialize 
	predictions, chosen_path, frames, e3nn_embs, vols, preds, num_coords = [], [], [], [], [], [], []
	pdb_id = '3MNP'
	paths = [x for x in os.walk(pdb_id)][0][2]
	
	# SORT PATHS
	sort_paths = []
	for index in range(len(paths)):
		for p in paths:
			if p == '.DS_Store':
				continue
			if index == int(p.split('.')[0].split('-')[-1]):
				sort_paths.append(p)
				continue 

	# ----------------------------- ITERATE THROUGH THE FRAMES -----------------------------------
	for idx, frame in enumerate(sort_paths):
		if frame == '.DS_Store':
			continue
		print(frame)
		#Apply fpocket
		# os.system(f'fpocket -f {pdb_id}/{frame} {fpocket_params}')
		# os.system(f'fpocket -f {pdb_id}/{frame}')
		# Get paths to pockets 
		f_no_pdb = frame.split('.')[0]
		pocket_paths = [pocket for pocket in 
					[i for i in os.walk(f'{pdb_id}/{f_no_pdb}_out/pockets')][0][2] if 'pdb' in pocket]
		
		path_order, frame_pockets = [], []
		
		# ----------------------------- ITERATE THROUGH THE POCKETS -----------------------------------
		first = False
		for pocket in pocket_paths:
			directory = f'{pdb_id}/{f_no_pdb}_out/pockets/{pocket}'
			if idx == 0 and first is True:
				continue

			# If is the first frame, we grab pocket 0
			if idx == 0:
				dir_0 = f'{pdb_id}/{f_no_pdb}_out/pockets/pocket0_atm.pdb'
				ppdb.read_pdb(dir_0)
				chosen_pocket = torch.FloatTensor(ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values)
				chosen_path.append(dir_0)
				vols.append(getPocketVolume(chosen_pocket))
				num_coords.append(chosen_pocket.shape[0])
				first = True
			
			# Otherwise we read and store all the pockets of the frame
			else:
				# Read pocket
				ppdb.read_pdb(directory)
				coords = torch.FloatTensor(ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values)
				frame_pockets.append(coords)
				path_order.append(directory)

		# Once we've read all the pockets within the same timestep, we generate centroids and calculate the distance
		# to the centroid of the previous one.
		if idx == 0:
			print(f'Pocket: {chosen_path[idx]}')
		else:
			print(f'Number of pocket identified by Fpocket: {len(frame_pockets)}')
			centroids = [getCentroid(pocket) for pocket in frame_pockets]
			distances = [calculateDistance(getCentroid(chosen_pocket), pcket_centroid) for pcket_centroid in centroids]
			# We grab the pocket of that holds the closest distance and we keep its path
			chosen_pocket = frame_pockets[distances.index(min(distances))]
			chosen_path.append(path_order[distances.index(min(distances))])
			print(f'New pocket: {chosen_path[idx]}')
			print(f'Minimum distance to previous pocket: {min(distances):.3f}')
			vol = getPocketVolume(chosen_pocket)
			print(f'Volume: {vol:.2f}\n')
			vols.append(vol)
			num_coords.append(chosen_pocket.shape[0])
	
		#----------------- EMBEDDINGS -----------------------
		pockets_set = buildDataset([chosen_pocket])
		pockets_loader = makeLoader(pockets_set)

		for batch in pockets_loader:
			e3nn_embedding = model(batch, embeddings = True)
			e3nn_embs.append(e3nn_embedding.detach().numpy())
			prediction = model(batch)
			preds.append(round(prediction.item()))
	
		
	#----------------- END OF FRAMES LOOP -----------------------
	print('\nE3NN embeddings:', np.squeeze(np.array(e3nn_embs)).shape)
	umap_embedding = reducer.fit_transform(np.squeeze(np.array(e3nn_embs)))
	print('UMAP Embeddings:',umap_embedding.shape,'\n')
	steps = np.arange(201)
	max_distance = 0
	shapes = ()
	# Get the two embeddings with the biggest distance between them.
	for drug_emb_idx in np.transpose(np.where(np.array(preds) > 0)):
		for nondrug_emb_idx in np.transpose(np.where(np.array(preds) == 0)):
			distance = calculateDistance(torch.Tensor(umap_embedding[drug_emb_idx]), torch.Tensor(umap_embedding[nondrug_emb_idx]))
			if distance > max_distance:
				max_distance = distance
				shapes = (drug_emb_idx[0], nondrug_emb_idx[0])

	print(f'Druggable: {chosen_path[shapes[0]]}, volume: {vols[shapes[0]]:.2f}\nNon druggable: {chosen_path[shapes[1]]}, volume: {vols[shapes[1]]:.2f}\nDistance: {max_distance:.2f}')
	# colors = cm.get_cmap('plasma')(np.linspace(0, 1, 201))
	colors = ['red' if i==0 else 'green' for i in preds]
	# Most 
	colors[shapes[0]] = 'blue'
	colors[shapes[1]] = 'blue'
	
	s = ['x' if i==0 else 'o' for i in preds]
	red_patch = mpatches.Patch(color='red', label='Non druggable')
	green_patch = mpatches.Patch(color='green', label='Druggable')
	x_min = umap_embedding[:,0].min()
	x_max = umap_embedding[:,0].max()
	y_min = umap_embedding[:,1].min()
	y_max = umap_embedding[:,1].max()

	#----------------------------------- VISUALIZATION - VOLUMES

	plt.figure(1, figsize = (12, 8))
	poly = np.polyfit(steps, vols, 12)
	poly_y = np.poly1d(poly)(steps)
	
	sns.lineplot(x = steps, y = poly_y, linewidth = 2.5)
	#plt.scatter(steps, vols, color = colors)
	sns.scatterplot(x = steps, y = vols, c = colors, style = s)

	plt.xlabel('Steps', fontsize = 14)
	plt.ylabel('Volume', fontsize = 14)
	plt.title(f'{pdb_id}')
	plt.legend(handles=[red_patch, green_patch], loc = 'best')
	plt.savefig(f'frame_images/{pdb_id}/Volumes{pdb_id}', dpi = 300)

	#----------------------------------- VISUALIZATION - NUMBER OF COORDS

	plt.figure(2, figsize = (12, 8))
	poly = np.polyfit(steps, num_coords, 12)
	poly_y = np.poly1d(poly)(steps)
	
	plt.plot(steps, poly_y, linewidth = 2.5)
	plt.scatter(steps, num_coords, color = colors)
	
	plt.xlabel('Steps', fontsize = 18)
	plt.ylabel('Number of coordinates', fontsize = 18)
	plt.title(f'{pdb_id}')
	plt.savefig(f'frame_images/{pdb_id}/Coords{pdb_id}', dpi = 300)

	#----------------------------------- VISUALIZATION - PREDICTIONS 

	plt.figure(3)
	sns.scatterplot(x = umap_embedding[:, 0], y = umap_embedding[:, 1], c = colors, style = s, alpha = 0.7)
	plt.xticks(np.arange(x_min, x_max))
	plt.yticks(np.arange(y_min, y_max))
	plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
	plt.xlabel('UMAP dimension 1', fontsize = 14)
	plt.ylabel('UMAP dimension 2', fontsize = 14)
	plt.legend(handles=[red_patch, green_patch], loc = 'best', prop={'size': 6})
	plt.savefig(f'frame_images/{pdb_id}/Predictions{pdb_id}', dpi = 300)

	#----------------------------------- VISUALIZATION - TIMESTEPS
	
	plt.figure(4)
	for idx, emb in enumerate(umap_embedding):
		plt.scatter(x = emb[0], y = emb[1], c = colors[idx], marker = s[idx], alpha = 0.7)
		plt.gca().set_aspect('equal', 'datalim')
		plt.xticks(np.arange(x_min, x_max))
		plt.yticks(np.arange(y_min, y_max))
		plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
		plt.xlabel('UMAP dimension 1', fontsize = 14)
		plt.ylabel('UMAP dimension 2', fontsize = 14)
		plt.savefig(f'frame_images/{pdb_id}/frames/frame{idx}', dpi= 300)
