import os 
import torch
from materials import Network, Convolution
from biopandas.pdb import PandasPdb


if __name__ == '__main__':
	ppdb = PandasPdb()
	# Initialize the model 
	model = Network()
	# model.load_state_dict(torch.load('./from/path/model.pt'))
	# model.eval()

	# Initialize 
	predictions, path_order, frames = [], [], []
	pdb_id = sys.argv([1])
	paths = [x for x in os.walk(pdb_id)][0][2]

	for frame in paths:
		if frame == '.DS_Store':
			continue
		_id = int(frame.split('.')[0].split('-')[-1])
		print('\nFrame:',_id)
		frames.append(_id)
		
		#Apply fpocket
		os.system(f'fpocket -f {pdb_id}/{frame}')
		# Get paths to pockets 
		f_no_pdb = frame.split('.')[0]
		pocket_paths = [pocket for pocket in 
					[i for i in os.walk(f'{pdb_id}/{f_no_pdb}_out/pockets')][0][2] if 'pdb' in pocket]
		
		# Iterate through the pockets 
		for pocket in pocket_paths:
			directory = f'{pdb_id}/{f_no_pdb}_out/pockets/{pocket}'
			# Save the path
			path_order.append(directory)
			# Get the coordinates of the pocket
			ppdb.read_pdb(directory)
			# Extract the coordinates of the pocket 
			coords = torch.tensor(ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values)
			# Predictions made by de model
			# pred = model(coords)
			# If model considers the pocket as druggable we print the path
			# if pred.round() == 1:
			# 	print(f'Druggable:{directory}')
			# predictions.append(pred.round())
	print(len(path_order))
		