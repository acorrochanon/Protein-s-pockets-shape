#FILE CONTAINS ALL THE FUNCTINS RELATED TO THE FPOCKET EXECUTION AND DATA RETRIEVAL. 

import os 
from copy import deepcopy
from common_funcs import *
# from funcs.common_funcs import *

# Extract all desired info of the pocket. Receives the path where the file is located. Returns coordinates, atom
# types, and features (polarity score...)
def extractFpocketInfo(path):
    coords, atom_types = [], []
    
    # Retrieve all data from pocket file
    data = [line for line in open(path)]
    
    # Get index of the line where the coordinates start
    indices, headers = [], []
    for idx, line in enumerate(data):
        if 'ATOM' in line or 'Information' in line:
            indices.append(idx)
        # HEADER required to know where the features end (3UFQ pocket 9 case)
        if 'HEADER' in line:
            headers.append(idx)
        # Only 2 indices wanted. Features starting point and end (it varies depending on fpocket version)
        if len(indices) == 2:
                break
    
    # Get pocket info (volume, Polarity Score...)
    pocket_info = [line.split()[-1] for line in data[(indices[0]+1):(headers[-1]+1)]]
    
    # [-2]= TER and END lines
    for atom in data[indices[1]:-2]:
        atom_line_split = atom.split()
        
        # There are format issues with some pocket .pdb files (mixed columns most of the times)
        # Extract coordinates. 
        for idx, col in enumerate(atom_line_split):
            if '.' in col:
                aux = []
                for value in atom_line_split[idx:]:
                    if len(value.split('.')) == 2:
                        aux.append(value)
                    
                    # 2 columns mixed
                    elif len(value.split('.')) == 3:
                        minus_split = value.split('-')
                        minus_indices = [idx for idx, char in enumerate(value) if char == '-']
                        if len(minus_indices) == 1:
                            aux.append(minus_split[0])
                            aux.append('-'+minus_split[1])
                        else:
                            aux.append('-'+minus_split[1])
                            aux.append('-'+minus_split[2])
                    
                    # 3 columns mixed    
                    elif len(value.split('.')) == 4:
                        minus_split = value.split('-')
                        minus_indices = [idx for idx, char in enumerate(value) if char == '-']
                        if len(minus_indices)==3:
                            aux.append('-'+minus_split[1])
                            aux.append('-'+minus_split[2])
                            aux.append('-'+minus_split[3])
                        # If there are 2 minus signs, first element cant be negative
                        else:
                            aux.append(minus_split[0])
                            aux.append('-'+minus_split[1])
                            aux.append('-'+minus_split[2])
                            

                    # Stop iterating when the x,y,z have been stored. 
                    if len(aux) == 3:
                        coords.append(aux)
                        break
                
                # We jump to the next line
                break
        
        # Retrieve atom types. Sometimes the atom type column is combined with the aminoacid chain. 
        if len(atom_line_split[2]) > 4:
            atom_types.append(atom_line_split[2][:-4])
        else:
            atom_types.append(atom_line_split[2])
    
    return torch.Tensor(np.array(coords, dtype='float32')), atom_types, pocket_info


# Calculates all the pockets for the given .pdb file and store them within the given folder name. 
def getFpockets(fname, txt_name, params = None):
    # Download PDB files first to allow fpocket be executed.
    # If folder is already created (1 is returned) we assume the pockets have been already generated. 
    # Note: If you want to extract pockets from a larger number proteins but you have already executed the method
    # previously, remove the folder and execute it again.
    
    pdb_path_list = []
    for pdb_path in os.walk(os.getcwd()+'/{}'.format(fname)):
        for pdb_file in pdb_path[2]:
            pdb_path_list.append('{}/{}'.format(fname, pdb_file))
    
    # Create txt with all .pdb paths
    txt_name = createTxt(pdb_path_list, fname)
    
    # Run command to extract all the pockets from each .PDB.
    if params is not None:
        # LOOP STYLE

        # with open(txt_name, 'r') as text_file:        
        #     for struct in text_file:
        #         os.system(f'fpocket {params} -f {struct} ')

        os.system(f'fpocket -F {txt_name} {params}')
    
    else:
        os.system(f'fpocket -F {txt_name}')
    
    # Remove txt file.
    os.system(f'rm {txt_name}')
    
    return 0


# Create and fill the structures that will handle all the data related to fpocket cavities. 
def generatePocketStructures(folder_pdb, pdb_dict):
    temp_pdb_dict = deepcopy(pdb_dict)
    # Auxiliary structures (protein-level)
    aux_pcoords, aux_patypes, aux_pinfo = [], [], []
    # Order in which pockets are read is mandatory. 
    path_order, pdb_order = [], []

    # Read files within the previously created folder. 
    for idx, path in enumerate(os.walk(os.getcwd()+'/'+folder_pdb)):
        # If 'pockets' is in the path it means current iteration is at some folder filled with pockets.
        if 'pockets' in path[0]:
            fpckt_coords, fpckt_atypes, fpckt_info = [], [], []
            
            # Pockets and files path
            pockets_path = path[0]
            files_path = path[2]

            # If we are at some duplicate folder, extract the coordinates from the original one (without the _'dup')
            if 'dup' in pockets_path.split('/')[-2]:
                pockets_path = path[0].replace('dup','out')
                files_path = [i[2] for i in os.walk(pockets_path)][0]
            
            # .pdb reading order within the PDB folder. Order is random. We need to keep track of them.
            pid = pockets_path.split('/')[-2].split('_')[0]
            pid_pos = temp_pdb_dict[pid][0]
            if pid_pos not in pdb_order:
                pdb_order.append(pid_pos)
                # Delete from pdb_ids
                temp_pdb_dict[pid].remove(pid_pos)
        
            # Iterate through the pockets
            for file in files_path:
                if '.pdb' in file:
                    # Order in which proteins and its pockets are read is random
                    # print('{}/{}'.format(path[0], file))
                    path_order.append('{}/{}'.format(pockets_path, file))
                    # Read pockets coord, atom types, and info 
                    pocket_coords, pocket_atypes, pocket_info = extractFpocketInfo('{}/{}'.format(pockets_path, file))
                    # Storage of all the pockets within a particular protein structure. Pocket-level                
                    fpckt_coords.append(pocket_coords)
                    fpckt_atypes.append(pocket_atypes)
                    fpckt_info.append(pocket_info)
    
            # Store the data for all the protein structures (protein-level)
            # index_PDB_ID X index_pocket x index_single_point
            aux_pcoords.append(fpckt_coords)
            aux_patypes.append(fpckt_atypes)
            aux_pinfo.append(fpckt_info)
    
    
    return aux_pcoords, aux_patypes, aux_pinfo, path_order, pdb_order